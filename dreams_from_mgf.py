"""
DREAMS Embeddings Extraction - Single MGF to Parquet

High-level flow:
1. Load MGF file with robust encoding detection and error handling
2. Generate DREAMS embeddings for all spectra in the file
3. Extract spectrum IDs from SCANS metadata field
4. Create DataFrame with spectrum_id and dreams_embedding columns
5. Save as parquet file for efficient storage and fast loading

Usage:
    python process_mgf_to_parquet.py <input_mgf> <output_parquet>
"""

import pandas as pd
from matchms.importing import load_from_mgf
from matchms import Spectrum
from dreams.api import dreams_embeddings
from tqdm import tqdm
import numpy as np
import os
import codecs
import chardet
import sys


def detect_encoding(file_path):
    """
    Detect the character encoding of a file by reading first 10KB.
    
    Args:
        file_path: Path to file to analyze
        
    Returns:
        String encoding name (e.g., 'utf-8', 'latin-1')
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']


def load_mgf_with_encoding(mgf_path):
    """
    Load MGF file with automatic encoding detection and fallback strategies.
    
    Tries multiple encoding methods:
    1. Auto-detected encoding
    2. Common encodings (utf-8, latin-1, cp1252)
    3. Manual line-by-line parsing with error ignoring
    
    Args:
        mgf_path: Path to MGF file
        
    Returns:
        List of matchms Spectrum objects
    """
    encoding = detect_encoding(mgf_path)
    print(f"  Detected encoding: {encoding}")
    
    # Try different encoding strategies in order of likelihood
    encodings_to_try = [encoding, 'utf-8', 'latin-1', 'utf-8-sig', 'cp1252']
    
    for enc in encodings_to_try:
        if enc is None:
            continue
        try:
            print(f"  Trying to load with {enc} encoding...")
            
            # Read with detected encoding, write normalized UTF-8 temp file
            temp_path = mgf_path + ".temp"
            with codecs.open(mgf_path, 'r', encoding=enc, errors='ignore') as f_in:
                content = f_in.read()
            
            with codecs.open(temp_path, 'w', encoding='utf-8') as f_out:
                f_out.write(content)
            
            # Load with matchms from normalized file
            spectra = list(load_from_mgf(temp_path, metadata_harmonization=False))
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"  ✓ Successfully loaded with {enc} encoding")
            return spectra
            
        except Exception as e:
            print(f"  Failed with {enc}: {str(e)[:100]}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
    
    # Final fallback: manual parsing with error handling
    print("  Attempting manual parsing with error handling...")
    return load_mgf_manually(mgf_path)


def load_mgf_manually(mgf_path):
    """
    Manually parse MGF file line-by-line with robust error handling.
    
    Parses MGF format:
    - BEGIN IONS / END IONS blocks
    - Metadata lines with KEY=VALUE format
    - Peak lines with m/z intensity pairs
    
    Args:
        mgf_path: Path to MGF file
        
    Returns:
        List of matchms Spectrum objects
    """
    spectra = []
    
    with open(mgf_path, 'r', encoding='utf-8', errors='ignore') as f:
        current_metadata = {}
        mz_list = []
        intensity_list = []
        in_ions = False
        
        for line in f:
            line = line.strip()
            
            if line == 'BEGIN IONS':
                in_ions = True
                current_metadata = {}
                mz_list = []
                intensity_list = []
                
            elif line == 'END IONS':
                if in_ions and len(mz_list) > 0:
                    # Create spectrum from accumulated data
                    spectrum = Spectrum(
                        mz=np.array(mz_list),
                        intensities=np.array(intensity_list),
                        metadata=current_metadata,
                        metadata_harmonization=False
                    )
                    spectra.append(spectrum)
                in_ions = False
                
            elif in_ions:
                if '=' in line:
                    # Metadata line: KEY=VALUE
                    key, value = line.split('=', 1)
                    current_metadata[key.lower()] = value
                elif line and not line.startswith('#'):
                    # Peak line: m/z intensity
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            mz_list.append(mz)
                            intensity_list.append(intensity)
                    except ValueError:
                        # Skip malformed peak lines
                        pass
    
    return spectra


def process_mgf_to_parquet(mgf_path, output_path):
    """
    Main processing pipeline: MGF → DREAMS embeddings → Parquet
    
    Pipeline steps:
    1. Validate input file exists
    2. Load all spectra from MGF with encoding handling
    3. Generate DREAMS embeddings for all spectra
    4. Extract spectrum IDs from SCANS metadata field
    5. Align IDs and embeddings into DataFrame
    6. Save as parquet file
    
    Args:
        mgf_path: Path to input MGF file
        output_path: Path for output parquet file
        
    Returns:
        Boolean indicating success/failure
    """
    
    print(f"\n{'='*60}")
    print(f"DREAMS EMBEDDINGS EXTRACTION")
    print(f"{'='*60}")
    print(f"Input MGF: {mgf_path}")
    print(f"Output parquet: {output_path}")
    
    # Validate input file
    if not os.path.exists(mgf_path):
        print(f"✗ ERROR: MGF file not found: {mgf_path}")
        return False
    
    file_size = os.path.getsize(mgf_path) / 1024**2
    print(f"MGF file size: {file_size:.2f} MB")
    print("-" * 50)
    
    # Step 1: Load spectra from MGF
    print(f"Loading spectra from MGF...")
    try:
        spectra = load_mgf_with_encoding(mgf_path)
        print(f"✓ Loaded {len(spectra)} spectra")
    except Exception as e:
        print(f"✗ ERROR loading MGF: {e}")
        return False
    
    # Verify spectrum count matches file
    print("Verifying spectrum count...")
    try:
        with open(mgf_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            begin_ions_count = content.count('BEGIN IONS')
        print(f"  Raw 'BEGIN IONS' count: {begin_ions_count}")
        print(f"  Loaded spectra count: {len(spectra)}")
        
        if len(spectra) != begin_ions_count:
            print(f"  ⚠ WARNING: Loaded {len(spectra)} but file has {begin_ions_count} BEGIN IONS markers")
    except Exception as e:
        print(f"  Could not verify counts: {e}")
    print("-" * 50)
    
    # Step 2: Generate DREAMS embeddings
    print(f"Generating DREAMS embeddings...")
    try:
        embs = dreams_embeddings(mgf_path)
        print(f"✓ Generated {len(embs)} embeddings")
        if len(embs) > 0:
            print(f"  Embedding shape: {embs[0].shape if hasattr(embs[0], 'shape') else len(embs[0]) if hasattr(embs[0], '__len__') else 'scalar'}")
    except Exception as e:
        print(f"✗ ERROR generating embeddings: {e}")
        return False
    print("-" * 50)
    
    # Step 3: Extract spectrum IDs from SCANS field
    print(f"Extracting spectrum IDs from SCANS field...")
    spectrum_ids = []
    missing_ids = 0
    
    # Show sample of available metadata keys
    if spectra:
        print(f"Sample metadata keys from first spectrum:")
        metadata_keys = list(spectra[0].metadata.keys())
        for key in metadata_keys[:10]:
            value = spectra[0].metadata[key]
            print(f"  {key}: {value}")
        print("-" * 50)
    
    # Extract IDs from SCANS field
    for i, spectrum in enumerate(tqdm(spectra, desc="Extracting spectrum IDs")):
        metadata = spectrum.metadata
        
        # Look for SCANS field (case-insensitive)
        spectrum_id = None
        scans_keys = ['scans', 'SCANS', 'scan', 'SCAN']
        
        for key in scans_keys:
            if key in metadata:
                spectrum_id = metadata[key]
                break
        
        # Fallback if no SCANS field found
        if spectrum_id is None or spectrum_id == '':
            missing_ids += 1
            spectrum_id = f"SPECTRUM_{i:08d}"
            if missing_ids == 1:
                print(f"\n  ⚠ No SCANS field found, using fallback: {spectrum_id}")
        
        spectrum_ids.append(str(spectrum_id))
        
        # Show first few samples
        if i < 3:
            print(f"  Sample {i}: spectrum_id={spectrum_id}")
    
    print(f"\n✓ Extracted {len(spectrum_ids)} spectrum IDs")
    if missing_ids > 0:
        print(f"  ⚠ {missing_ids} spectra had no SCANS field")
    print("-" * 50)
    
    # Step 4: Align data - ensure same length
    print("Aligning data...")
    print(f"  Spectra count: {len(spectra)}")
    print(f"  IDs count: {len(spectrum_ids)}")
    print(f"  Embeddings count: {len(embs)}")
    
    if len(spectrum_ids) < len(embs):
        print(f"  Adding {len(embs) - len(spectrum_ids)} placeholder IDs...")
        while len(spectrum_ids) < len(embs):
            spectrum_ids.append(f"PLACEHOLDER_{len(spectrum_ids):08d}")
    elif len(spectrum_ids) > len(embs):
        print(f"  Truncating IDs to match embeddings...")
        spectrum_ids = spectrum_ids[:len(embs)]
    
    print(f"  Final alignment: {len(spectrum_ids)} IDs, {len(embs)} embeddings")
    print("-" * 50)
    
    # Step 5: Create DataFrame
    print(f"Creating DataFrame...")
    df = pd.DataFrame({
        'spectrum_id': spectrum_ids,
        'dreams_embedding': list(embs)
    })
    print(f"✓ DataFrame shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Step 6: Save as parquet
    print(f"Saving to parquet...")
    try:
        df.to_parquet(output_path, index=False)
        print(f"✓ Saved {len(df)} spectra with embeddings")
        
        # Verify saved file
        df_verify = pd.read_parquet(output_path)
        print(f"✓ Verification successful: {df_verify.shape} rows")
        print(f"  Columns in saved file: {df_verify.columns.tolist()}")
        
        output_size = os.path.getsize(output_path) / 1024**2
        print(f"  Output file size: {output_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ ERROR saving: {e}")
        return False
    
    print("="*60)
    return True


if __name__ == "__main__":
    # Install chardet if needed for encoding detection
    try:
        import chardet
    except ImportError:
        print("Installing chardet for encoding detection...")
        import subprocess
        subprocess.check_call(["pip", "install", "chardet"])
        import chardet
    
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python process_mgf_to_parquet.py <input_mgf> <output_parquet>")
        print("\nExample:")
        print("  python process_mgf_to_parquet.py data.mgf embeddings.parquet")
        sys.exit(1)
    
    mgf_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Ensure output has .parquet extension
    if not output_path.endswith('.parquet'):
        output_path += '.parquet'
    
    print("="*60)
    print("DREAMS EMBEDDINGS EXTRACTION")
    print("MGF → Parquet with spectrum_id and dreams_embedding")
    print("="*60)
    
    # Process the file
    success = process_mgf_to_parquet(mgf_path, output_path)
    
    if success:
        print("\n✅ PROCESSING COMPLETED SUCCESSFULLY!")
    else:
        print("\n✗ PROCESSING FAILED")
        sys.exit(1)
        