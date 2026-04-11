"""
Data Preprocessing Module
Handles NIfTI file reading, normalization, and slice extraction for BraTS dataset
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Union, List, Dict
import logging

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_DEPTH,
    NORMALIZATION_METHOD,
    INTENSITY_CLIP_RANGE,
    MRI_MODALITIES,
)

logger = logging.getLogger(__name__)


class NIfTIPreprocessor:
    """Handles NIfTI file preprocessing including normalization and resizing"""

    def __init__(
        self,
        height: int = IMAGE_HEIGHT,
        width: int = IMAGE_WIDTH,
        depth: int = IMAGE_DEPTH,
        normalization_method: str = NORMALIZATION_METHOD,
        intensity_clip_range: Tuple[float, float] = INTENSITY_CLIP_RANGE,
    ):
        """
        Initialize preprocessor

        Args:
            height: Target height for images
            width: Target width for images
            depth: Target depth (number of slices)
            normalization_method: 'zscore' or 'minmax'
            intensity_clip_range: Clipping range after normalization
        """
        self.height = height
        self.width = width
        self.depth = depth
        self.normalization_method = normalization_method
        self.intensity_clip_range = intensity_clip_range

    @staticmethod
    def load_nifti(nifti_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load NIfTI file and extract metadata

        Args:
            nifti_path: Path to .nii or .nii.gz file

        Returns:
            Tuple of (image_array, metadata_dict)
        """
        nifti_path = Path(nifti_path)
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

        # Load NIfTI file
        nifti_img = nib.load(nifti_path)
        image_data = nifti_img.get_fdata()

        # Extract metadata
        metadata = {
            "affine": nifti_img.affine,
            "header": nifti_img.header,
            "shape": image_data.shape,
            "dtype": image_data.dtype,
        }

        logger.debug(f"Loaded NIfTI: {nifti_path.name}, shape={image_data.shape}")
        return image_data, metadata

    @staticmethod
    def save_nifti(
        image_array: np.ndarray,
        output_path: Union[str, Path],
        affine: np.ndarray = None,
    ) -> None:
        """
        Save array as NIfTI file

        Args:
            image_array: Image data array
            output_path: Output file path
            affine: Affine transformation matrix (identity if None)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if affine is None:
            affine = np.eye(4)

        nifti_img = nib.Nifti1Image(image_array, affine)
        nib.save(nifti_img, output_path)
        logger.debug(f"Saved NIfTI: {output_path.name}")

    def normalize_zscore(
        self, image: np.ndarray, epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Z-score normalization (standardization)
        Formula: (x - mean) / (std + epsilon)

        Args:
            image: Input image array
            epsilon: Small value to avoid division by zero

        Returns:
            Normalized image
        """
        mean = image.mean()
        std = image.std()
        normalized = (image - mean) / (std + epsilon)
        return normalized

    def normalize_minmax(self, image: np.ndarray) -> np.ndarray:
        """
        Min-max normalization (rescale to [0, 1])
        Formula: (x - min) / (max - min)

        Args:
            image: Input image array

        Returns:
            Normalized image in [0, 1] range
        """
        img_min = image.min()
        img_max = image.max()

        if img_max == img_min:
            return np.zeros_like(image)

        normalized = (image - img_min) / (img_max - img_min)
        return normalized

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply normalization based on configured method

        Args:
            image: Input image array

        Returns:
            Normalized image
        """
        if self.normalization_method == "zscore":
            normalized = self.normalize_zscore(image)
        elif self.normalization_method == "minmax":
            normalized = self.normalize_minmax(image)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        # Clip to range
        normalized = np.clip(normalized, self.intensity_clip_range[0], self.intensity_clip_range[1])
        return normalized

    def crop_center(
        self, image: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Crop image from center to target shape

        Args:
            image: Input image array
            target_shape: Target shape (D, H, W)

        Returns:
            Cropped image
        """
        slices = []
        for i, (dim_size, target_size) in enumerate(zip(image.shape, target_shape)):
            if dim_size <= target_size:
                slices.append(slice(0, dim_size))
            else:
                start = (dim_size - target_size) // 2
                slices.append(slice(start, start + target_size))

        return image[tuple(slices)]

    def pad_to_size(
        self, image: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Pad image to target shape using zeros

        Args:
            image: Input image array
            target_shape: Target shape (D, H, W)

        Returns:
            Padded image
        """
        pad_widths = []
        for i, (current_size, target_size) in enumerate(zip(image.shape, target_shape)):
            if current_size >= target_size:
                pad_widths.append((0, 0))
            else:
                pad_total = target_size - current_size
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                pad_widths.append((pad_before, pad_after))

        return np.pad(image, pad_widths, mode="constant", constant_values=0)

    def resize_3d(
        self, image: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Resize 3D image to target shape (crop or pad as needed)

        Args:
            image: Input image array (D, H, W)
            target_shape: Target shape (D, H, W)

        Returns:
            Resized image
        """
        if image.shape == target_shape:
            return image

        # First crop if necessary
        crop_shape = tuple(
            min(s, t) for s, t in zip(image.shape, target_shape)
        )
        image = self.crop_center(image, crop_shape)

        # Then pad if necessary
        image = self.pad_to_size(image, target_shape)
        return image

    def preprocess(
        self,
        image: np.ndarray,
        resample: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Apply complete preprocessing pipeline

        Args:
            image: Input image array
            resample: Whether to resize to standard shape
            normalize: Whether to apply normalization

        Returns:
            Preprocessed image
        """
        # Resize to standard shape
        if resample:
            image = self.resize_3d(image, (self.depth, self.height, self.width))

        # Normalize
        if normalize:
            image = self.normalize(image)

        return image


class BraTS2021Loader:
    """
    Handler for BraTS 2021 dataset structure
    Expected directory structure:
        data/raw/BRATS/
            ├── BraTS2021_<split>/
            │   ├── BraTS2021_{id}/
            │   │   ├── BraTS2021_{id}_t1.nii.gz
            │   │   ├── BraTS2021_{id}_t1ce.nii.gz
            │   │   ├── BraTS2021_{id}_t2.nii.gz
            │   │   ├── BraTS2021_{id}_flair.nii.gz
            │   │   └── BraTS2021_{id}_seg.nii.gz
    """

    MODALITIES = MRI_MODALITIES
    SEGMENTATION_FILE = "seg"

    def __init__(self, data_root: Union[str, Path]):
        """
        Initialize BraTS loader

        Args:
            data_root: Root directory containing BraTS dataset
        """
        self.data_root = Path(data_root)
        self.preprocessor = NIfTIPreprocessor()

    def get_case_files(self, case_id: str, split: str = "Training") -> Dict[str, Path]:
        """
        Get all file paths for a specific case

        Args:
            case_id: Patient ID (e.g., "00000")
            split: Dataset split ("Training", "Validation", "Testing")

        Returns:
            Dictionary mapping modality to file path
        """
        case_dir = self.data_root / f"BraTS2021_{split}" / f"BraTS2021_{case_id}"

        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        files = {}
        for modality in self.MODALITIES:
            modality_code = modality.lower() if modality != "T1ce" else "t1ce"
            file_path = case_dir / f"BraTS2021_{case_id}_{modality_code}.nii.gz"

            if not file_path.exists():
                raise FileNotFoundError(f"Modality file not found: {file_path}")

            files[modality] = file_path

        # Add segmentation
        seg_path = case_dir / f"BraTS2021_{case_id}_{self.SEGMENTATION_FILE}.nii.gz"
        if seg_path.exists():
            files["segmentation"] = seg_path
        else:
            logger.warning(f"Segmentation file not found for {case_id}")

        return files

    def load_case(
        self,
        case_id: str,
        split: str = "Training",
        preprocess: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Load all modalities and segmentation for a case

        Args:
            case_id: Patient ID
            split: Dataset split
            preprocess: Whether to apply preprocessing

        Returns:
            Dictionary with modalities and segmentation
        """
        files = self.get_case_files(case_id, split)

        # Load modalities
        modalities = {}
        for modality, file_path in files.items():
            if modality == "segmentation":
                continue

            image, metadata = NIfTIPreprocessor.load_nifti(file_path)

            if preprocess:
                image = self.preprocessor.preprocess(image)

            modalities[modality] = image

        # Stack modalities
        stacked = np.stack(
            [modalities[mod] for mod in self.MODALITIES],
            axis=0
        )  # (C, D, H, W)

        result = {"images": stacked}

        # Load segmentation if available
        if "segmentation" in files:
            seg, _ = NIfTIPreprocessor.load_nifti(files["segmentation"])
            if preprocess:
                seg = self.preprocessor.preprocess(seg, normalize=False)
            result["segmentation"] = seg

        return result

    def get_all_case_ids(self, split: str = "Training") -> List[str]:
        """
        Get all case IDs in a split

        Args:
            split: Dataset split

        Returns:
            List of case IDs
        """
        split_dir = self.data_root / f"BraTS2021_{split}"

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return []

        case_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        case_ids = [
            d.replace("BraTS2021_", "") for d in case_dirs
        ]
        return case_ids


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test NIfTI loading and preprocessing
    preprocessor = NIfTIPreprocessor()

    # Test normalization
    test_image = np.random.randn(155, 240, 240)
    normalized = preprocessor.preprocess(test_image)
    print(f"Original shape: {test_image.shape}, dtype: {test_image.dtype}")
    print(f"Normalized shape: {normalized.shape}, dtype: {normalized.dtype}")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    # Test BraTS loader
    # brats_loader = BraTS2021Loader("data/BraTS")
    # case_ids = brats_loader.get_all_case_ids("Training")
    # print(f"Found {len(case_ids)} cases in Training split")
