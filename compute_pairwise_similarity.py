"""
Compute pairwise cosine similarity from DreaMS embeddings using FAISS.

Reads embeddings.parquet, finds all pairs with cosine similarity >= THRESHOLD,
outputs edges.parquet with columns (spectrum_id_1, spectrum_id_2, cosine_similarity).

Two modes:
  MODE = "gpu"  → GPU top-K search + threshold filter (fast, needs TOP_K large enough)
  MODE = "cpu"  → CPU range_search (exact, slower)
"""

import time
import argparse
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

def build_index(embeddings, mode):
    """Build FAISS index. GPU mode tries all GPUs, falls back to CPU on failure."""
    D = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(D)

    if mode == "gpu":
        for gpu_id in range(faiss.get_num_gpus()):
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                gpu_index.add(embeddings)
                gpu_index.search(embeddings[:1], 2)
                print(f"  Using GPU {gpu_id}")
                return gpu_index
            except Exception as e:
                print(f"  GPU {gpu_id} failed: {e}")
        print("  No GPU available, falling back to CPU")

    cpu_index.add(embeddings)
    print("  Using CPU")
    return cpu_index


def search_gpu(index, embeddings, spectrum_ids, threshold, top_k, batch_size):
    """GPU top-K search + threshold filter."""
    N = len(embeddings)
    K = min(top_k, N)
    all_id1, all_id2, all_sim = [], [], []
    total_edges = 0
    truncated = 0

    pbar = tqdm(total=N, desc="GPU search", unit="spectra")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        scores, neighbors = index.search(embeddings[start:end], K)

        for i_in_batch in range(end - start):
            global_i = start + i_in_batch
            nbrs = neighbors[i_in_batch]
            sims = scores[i_in_batch]

            if sims[-1] >= threshold:
                truncated += 1

            mask = (sims >= threshold) & (nbrs > global_i)
            j_indices = nbrs[mask]
            j_scores = sims[mask]

            if len(j_indices) > 0:
                all_id1.append(np.full(len(j_indices), spectrum_ids[global_i]))
                all_id2.append(spectrum_ids[j_indices])
                all_sim.append(j_scores)
                total_edges += len(j_indices)

        pbar.update(end - start)
        pbar.set_postfix(edges=total_edges)
    pbar.close()

    if truncated > 0:
        print(f"  WARNING: {truncated} spectra may have more than K={K} neighbors above threshold. Consider increasing TOP_K.")

    return all_id1, all_id2, all_sim, total_edges


def search_cpu(index, embeddings, spectrum_ids, threshold, batch_size):
    """CPU range_search (exact)."""
    N = len(embeddings)
    all_id1, all_id2, all_sim = [], [], []
    total_edges = 0

    pbar = tqdm(total=N, desc="CPU range_search", unit="spectra")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        lims, D_scores, I = index.range_search(embeddings[start:end], threshold)

        for i_in_batch in range(end - start):
            global_i = start + i_in_batch
            neighbors = I[lims[i_in_batch]:lims[i_in_batch + 1]]
            scores = D_scores[lims[i_in_batch]:lims[i_in_batch + 1]]

            mask = neighbors > global_i
            j_indices = neighbors[mask]
            j_scores = scores[mask]

            if len(j_indices) > 0:
                all_id1.append(np.full(len(j_indices), spectrum_ids[global_i]))
                all_id2.append(spectrum_ids[j_indices])
                all_sim.append(j_scores)
                total_edges += len(j_indices)

        pbar.update(end - start)
        pbar.set_postfix(edges=total_edges)
    pbar.close()

    return all_id1, all_id2, all_sim, total_edges


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarity from DreaMS embeddings using FAISS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required input file
  python compute_pairwise_similarity.py input.parquet
  
  # Specify output file and threshold
  python compute_pairwise_similarity.py input.parquet -o output.parquet -t 0.9
  
  # Use CPU mode with custom batch size
  python compute_pairwise_similarity.py input.parquet -m cpu -b 50000
  
  # GPU mode with custom top-k
  python compute_pairwise_similarity.py input.parquet -m gpu -k 4096
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input parquet file path containing embeddings (must have 'spectrum_id' and 'dreams_embedding' columns)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output parquet file path (default: input filename with '_edges' suffix)"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold (default: 0.8)"
    )
    
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Search mode: 'gpu' for GPU top-K search (fast) or 'cpu' for CPU range_search (exact) (default: gpu)"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=2048,
        help="(GPU only) Maximum neighbors per query. Increase if truncation warning appears (default: 2048)"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=8192,
        help="Batch size for processing. GPU default: 8192, CPU default: 50000 (default: 8192)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set output path if not provided
    if args.output is None:
        if args.input.endswith('.parquet'):
            args.output = args.input.replace('.parquet', '_edges.parquet')
        else:
            args.output = args.input + '_edges.parquet'
    
    # Adjust batch size for CPU mode
    if args.mode == "cpu":
        batch_size = max(args.batch_size, 50000)
    else:
        batch_size = args.batch_size
    
    t0 = time.time()

    # 1. Load embeddings
    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    spectrum_ids = df["spectrum_id"].values
    embeddings = np.stack(df["dreams_embedding"].values).astype(np.float32)
    N, D = embeddings.shape
    print(f"  {N} spectra, {D}-dim embeddings, {embeddings.nbytes / 1e9:.2f} GB")

    # 2. L2 normalize → inner product = cosine similarity
    faiss.normalize_L2(embeddings)

    # 3. Build index
    print(f"Building FAISS IndexFlatIP (mode={args.mode}) ...")
    index = build_index(embeddings, args.mode)
    print(f"  Index built with {index.ntotal} vectors")

    # 4. Search
    if args.mode == "gpu":
        print(f"Running GPU top-K search (K={args.top_k}, threshold={args.threshold}) ...")
        all_id1, all_id2, all_sim, total_edges = search_gpu(
            index, embeddings, spectrum_ids, args.threshold, args.top_k, batch_size)
    else:
        print(f"Running CPU range_search (threshold={args.threshold}) ...")
        all_id1, all_id2, all_sim, total_edges = search_cpu(
            index, embeddings, spectrum_ids, args.threshold, batch_size)

    # 5. Save
    print(f"Saving {total_edges} edges to {args.output} ...")
    if total_edges > 0:
        edges_df = pd.DataFrame({
            "spectrum_id_1": np.concatenate(all_id1),
            "spectrum_id_2": np.concatenate(all_id2),
            "cosine_similarity": np.concatenate(all_sim).astype(np.float32),
        })
    else:
        edges_df = pd.DataFrame(columns=["spectrum_id_1", "spectrum_id_2", "cosine_similarity"])

    edges_df.to_parquet(args.output, index=False)

    elapsed = time.time() - t0
    print(f"Done. {total_edges} edges, {elapsed:.1f}s elapsed.")
    print(f"Output: {args.output} ({edges_df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory)")


if __name__ == "__main__":
    main()
