"""
Script to download and prepare BraTS dataset
Requires registration at https://www.med.upenn.edu/cbica/brats2021/
"""

import argparse
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def download_brats(year: int = 2021, download_dir: str = "data/BraTS"):
    """
    Guide for downloading BraTS dataset

    Args:
        year: BraTS year (2021, 2020, 2019, etc.)
        download_dir: Directory to store dataset
    """
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"BraTS {year} Dataset Download Guide")
    logger.info("=" * 80)

    instructions = f"""

    The BraTS (Brain Tumor Segmentation) dataset requires manual registration
    and download due to medical data privacy regulations.

    STEPS TO DOWNLOAD:

    1. Visit: https://www.med.upenn.edu/cbica/brats{year}/

    2. Register for an account (if you don't have one)

    3. Download the dataset files

    4. Extract the dataset to: {download_path.absolute()}

    EXPECTED DIRECTORY STRUCTURE:

    {download_path.absolute()}/
    ├── BraTS{year}_Training/
    │   ├── BraTS{year}_00000/
    │   │   ├── BraTS{year}_00000_t1.nii.gz
    │   │   ├── BraTS{year}_00000_t1ce.nii.gz
    │   │   ├── BraTS{year}_00000_t2.nii.gz
    │   │   ├── BraTS{year}_00000_flair.nii.gz
    │   │   └── BraTS{year}_00000_seg.nii.gz
    │   ├── BraTS{year}_00001/
    │   │   └── ...
    │   └── ...
    ├── BraTS{year}_Validation/
    │   └── ...
    └── BraTS{year}_Testing/
        └── ...

    TROUBLESHOOTING:

    - If you receive a "File not found" error, verify the directory structure above
    - Case IDs should be zero-padded (00000, 00001, etc.)
    - Modality codes should be lowercase: t1, t1ce, t2, flair
    - Segmentation file should be named: *_seg.nii.gz

    DATASET STATISTICS (BraTS 2021):

    - Training: 369 cases (with segmentation labels)
    - Validation: 125 cases (with segmentation labels)
    - Testing: 219 cases (without segmentation labels)
    - Total size: ~150 GB (uncompressed)

    ALTERNATIVE: Using preprocessed data

    If you don't have access to the raw dataset, you can:
    1. Use a smaller medical imaging dataset (e.g., from a public challenge)
    2. Create synthetic multimodal data for testing
    3. Work with sample 2D slices (faster for development)

    For testing purposes, you can create mock data:
        python scripts/create_mock_data.py --num_cases 5
    """

    logger.info(instructions)

    # Create a README for the download directory
    readme_path = download_path / "DOWNLOAD_INSTRUCTIONS.md"
    with open(readme_path, "w") as f:
        f.write(instructions)

    logger.info(f"Instructions saved to: {readme_path}")


def validate_dataset(data_dir: str = "data/BraTS", year: int = 2021):
    """
    Validate downloaded dataset structure

    Args:
        data_dir: Directory containing dataset
        year: BraTS year
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Dataset directory not found: {data_path}")
        return False

    # Check for expected directories
    expected_splits = ["Training", "Validation", "Testing"]
    found_splits = []

    for split in expected_splits:
        split_dir = data_path / f"BraTS{year}_{split}"
        if split_dir.exists():
            found_splits.append(split)
            case_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            logger.info(f"  {split}: {len(case_dirs)} cases")
        else:
            logger.warning(f"  {split}: NOT FOUND")

    if len(found_splits) == 0:
        logger.error("No BraTS splits found. Please download the dataset.")
        return False

    logger.info(f"Dataset validation complete. Found {len(found_splits)}/3 splits.")
    return True


def create_mock_data(
    num_cases: int = 5,
    output_dir: str = "data/BraTS_mock",
    year: int = 2021,
):
    """
    Create mock BraTS data for testing (without downloading full dataset)

    Args:
        num_cases: Number of mock cases to create
        output_dir: Output directory for mock data
        year: BraTS year
    """
    import numpy as np
    import nibabel as nib

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_dir = output_path / f"BraTS{year}_Training"
    split_dir.mkdir(exist_ok=True)

    modalities = ["t1", "t1ce", "t2", "flair"]

    logger.info(f"Creating {num_cases} mock cases...")

    for case_id in range(num_cases):
        case_id_str = f"{case_id:05d}"
        case_dir = split_dir / f"BraTS{year}_{case_id_str}"
        case_dir.mkdir(exist_ok=True)

        # Create mock modality volumes
        for modality in modalities:
            # Random 3D volume
            volume = np.random.randn(155, 240, 240).astype(np.float32) * 50 + 100
            volume = np.clip(volume, 0, 255)

            # Save as NIfTI
            nifti_img = nib.Nifti1Image(volume, np.eye(4))
            nifti_path = case_dir / f"BraTS{year}_{case_id_str}_{modality}.nii.gz"
            nib.save(nifti_img, nifti_path)

        # Create mock segmentation
        seg = np.zeros((155, 240, 240), dtype=np.uint8)

        # Add some random tumor regions
        tumor_mask = np.random.rand(155, 240, 240) > 0.95  # ~5% tumor
        seg[tumor_mask] = np.random.randint(1, 4)  # Classes 1-3

        nifti_seg = nib.Nifti1Image(seg, np.eye(4))
        seg_path = case_dir / f"BraTS{year}_{case_id_str}_seg.nii.gz"
        nib.save(nifti_seg, seg_path)

        logger.info(f"  Created case: {case_id_str}")

    logger.info(f"Mock data created in: {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="BraTS Dataset Download Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python scripts/download_data.py --year 2021

  # Validate existing dataset
  python scripts/download_data.py --validate --data_dir data/BraTS

  # Create mock data for testing
  python scripts/download_data.py --create_mock --num_cases 5
        """,
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2021,
        help="BraTS year (default: 2021)",
    )

    parser.add_argument(
        "--data_dir",
        "--output_dir",
        dest="data_dir",
        type=str,
        default="data/BraTS",
        help="Dataset/output directory (default: data/BraTS)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset",
    )

    parser.add_argument(
        "--create_mock",
        action="store_true",
        help="Create mock dataset for testing",
    )

    parser.add_argument(
        "--num_cases",
        type=int,
        default=5,
        help="Number of mock cases to create (default: 5)",
    )

    args = parser.parse_args()

    if args.validate:
        success = validate_dataset(args.data_dir, args.year)
        sys.exit(0 if success else 1)

    elif args.create_mock:
        try:
            create_mock_data(args.num_cases, args.data_dir, args.year)
            logger.info("Mock data creation complete!")
        except Exception as e:
            logger.error(f"Error creating mock data: {e}")
            sys.exit(1)

    else:
        download_brats(args.year, args.data_dir)


if __name__ == "__main__":
    main()
