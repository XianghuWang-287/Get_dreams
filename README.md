# To generate DreaMS embeddings:


``` 
# if DreaMS doesn't exist in this directory yet
git clone https://github.com/pluskal-lab/DreaMS.git 
cd DreaMS

# Create conda environment
conda create -n dreams_env python==3.11.0 --yes
conda activate dreams_env

# Install DreaMS
pip install -e .
```

Then 
```
cd ..
pip install chardet
python3 dreams_from_mgf.py input_mgf_path.mgf embeddings_output.parquet
```

Then DreaMS will download its embedding model if it isnt downloaded yet.


## Run it in docker

```
make build-docker
```

Then you can run the container with:

```
cd data && sh ./get_data.sh
make run-docker
``` 

### Hello world for the script in docker
```
source activate dreams_env

python ./dreams_from_mgf.py /data/astral_water_and_methanol_spectra_reformatted.mgf /dev/null

#python3 dreams_from_mgf.py input_mgf_path.mgf embeddings_output.parquet
```

---

## Compute Pairwise Cosine Similarity

`compute_pairwise_similarity.py` reads a DreaMS embeddings parquet file, computes all pairwise cosine similarities above a threshold using FAISS, and outputs an edge list.

### Input Format

The input `.parquet` file must contain two columns:
- `spectrum_id` — unique identifier for each spectrum
- `dreams_embedding` — the DreaMS embedding vector

### Output Format

The output `.parquet` file contains three columns:
- `spectrum_id_1` — first spectrum ID
- `spectrum_id_2` — second spectrum ID
- `cosine_similarity` — cosine similarity score (≥ threshold)

### Usage

```
python compute_pairwise_similarity.py <input.parquet> [options]
```

### Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `input` | (positional) | *required* | Input parquet file with embeddings |
| `--output` | `-o` | `<input>_edges.parquet` | Output parquet file path |
| `--threshold` | `-t` | `0.8` | Cosine similarity threshold |
| `--mode` | `-m` | `gpu` | `gpu` (fast, top-K) or `cpu` (exact, range search) |
| `--top-k` | `-k` | `2048` | (GPU only) Max neighbors per query |
| `--batch-size` | `-b` | `8192` | Batch size for processing |

### Examples

```bash
# Basic usage (GPU mode, threshold=0.8)
python compute_pairwise_similarity.py embeddings.parquet

# Custom output path and threshold
python compute_pairwise_similarity.py embeddings.parquet -o results.parquet -t 0.9

# CPU mode (exact search, slower but no truncation)
python compute_pairwise_similarity.py embeddings.parquet -m cpu

# GPU mode with larger top-k (if truncation warning appears)
python compute_pairwise_similarity.py embeddings.parquet -k 4096
```

### Run in Docker

1. Build the Docker image:
```bash
make build-docker
```

2. Start the container with GPU support:
```bash
make run-docker-gpu
```

3. Inside the container, activate the environment and run:
```bash
source activate dreams_env
python3 ./compute_pairwise_similarity.py ./embeddings.parquet
```

Full example with all options:
```bash
source activate dreams_env
python3 ./compute_pairwise_similarity.py /data/embeddings.parquet -o /data/edges.parquet -t 0.8 -m gpu -k 2048 -b 8192
```

> **Note:** Use `make run-docker-gpu` (not `make run-docker`) to enable GPU access inside the container. The `--gpus all` flag is required for FAISS GPU mode.