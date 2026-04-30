"""
Combine separate BraTS channel files into a single 4-channel NIfTI file

BraTS files are typically stored separately:
- patient_t1.nii (T1-weighted MRI)
- patient_t1ce.nii (T1-weighted contrast-enhanced)
- patient_t2.nii (T2-weighted)
- patient_flair.nii (FLAIR)

This script combines them into a single 4-channel file:
- patient_combined.nii.gz (H, W, D, 4)

Usage:
    python scripts/combine_brats_files.py --input_dir BraTS19_014 --output combined.nii.gz
    python scripts/combine_brats_files.py --input_dir data/BraTS_Training/BraTS19_001
"""

import argparse
import logging
from pathlib import Path
import nibabel as nib
import numpy as np
from typing import Tuple, Union, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_brats_files(input_dir: Path) -> dict:
    """
    Find T1, T1ce, T2, FLAIR files in directory
    
    Args:
        input_dir: Directory containing BraTS files
        
    Returns:
        Dictionary with keys: 't1', 't1ce', 't2', 'flair'
    """
    files = {}
    
    # Look for standard naming patterns
    patterns = {
        't1': ['*_t1.nii', '*_t1.nii.gz', '*t1.nii', '*t1.nii.gz'],
        't1ce': ['*_t1ce.nii', '*_t1ce.nii.gz', '*t1ce.nii', '*t1ce.nii.gz'],
        't2': ['*_t2.nii', '*_t2.nii.gz', '*t2.nii', '*t2.nii.gz'],
        'flair': ['*_flair.nii', '*_flair.nii.gz', '*flair.nii', '*flair.nii.gz'],
    }
    
    for modality, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = list(input_dir.glob(pattern))
            if matches:
                files[modality] = matches[0]
                logger.info(f"Found {modality:6} → {matches[0].name}")
                break
        
        if modality not in files:
            raise FileNotFoundError(f"Could not find {modality} file in {input_dir}")
    
    return files


def load_nifti(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file and return data + affine matrix"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def combine_brats_files(input_dir: Union[str, Path], output_path: Optional[str] = None, verbose: bool = True) -> str:
    """
    Combine separate BraTS channel files into single 4-channel file
    
    Args:
        input_dir: Directory containing T1, T1ce, T2, FLAIR files
        output_path: Output file path (default: {input_dir}_combined.nii.gz)
        verbose: Print progress information
        
    Returns:
        Path to output file
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if verbose:
        logger.info(f"\n{'='*70}")
        logger.info(f"Combining BraTS files from: {input_dir.name}")
        logger.info(f"{'='*70}\n")
    
    # Find files
    files = find_brats_files(input_dir)
    
    # Load files
    if verbose:
        logger.info("Loading NIfTI files...")
    
    t1_data, affine = load_nifti(files['t1'])
    t1ce_data, _ = load_nifti(files['t1ce'])
    t2_data, _ = load_nifti(files['t2'])
    flair_data, _ = load_nifti(files['flair'])
    
    if verbose:
        logger.info(f"T1:    {t1_data.shape}")
        logger.info(f"T1ce:  {t1ce_data.shape}")
        logger.info(f"T2:    {t2_data.shape}")
        logger.info(f"FLAIR: {flair_data.shape}")
    
    # Verify shapes match
    if not (t1_data.shape == t1ce_data.shape == t2_data.shape == flair_data.shape):
        raise ValueError("Channel shapes don't match!")
    
    # Stack channels: (H, W, D, 4)
    if verbose:
        logger.info("\nStacking channels...")
    
    combined = np.stack([t1_data, t1ce_data, t2_data, flair_data], axis=-1)
    
    if verbose:
        logger.info(f"Combined shape: {combined.shape}")
    
    # Save as NIfTI
    if output_path is None:
        output_file: Path = Path(f"{input_dir.name}_combined.nii.gz")
    else:
        output_file = Path(output_path)
    
    if verbose:
        logger.info(f"\nSaving to: {output_file}")
    
    img = nib.Nifti1Image(combined, affine=affine)
    nib.save(img, output_file)
    
    if verbose:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Success! File size: {file_size_mb:.1f} MB")
        logger.info(f"\nYou can now upload this file to the app:")
        logger.info(f"  → {output_file.absolute()}")
        logger.info(f"{'='*70}\n")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Combine separate BraTS channel files into single 4-channel NIfTI file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine files from BraTS19_014 folder in current directory
  python scripts/combine_brats_files.py --input_dir BraTS19_014
  
  # Combine and specify output name
  python scripts/combine_brats_files.py --input_dir BraTS19_014 --output my_scan.nii.gz
  
  # Combine from nested path
  python scripts/combine_brats_files.py --input_dir data/BraTS_Training/BraTS19_001
        """,
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing T1, T1ce, T2, FLAIR .nii files',
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: {input_dir}_combined.nii.gz)',
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output',
    )
    
    args = parser.parse_args()
    
    try:
        output_file = combine_brats_files(
            args.input_dir,
            args.output,
            verbose=not args.quiet,
        )
        print(output_file)
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
