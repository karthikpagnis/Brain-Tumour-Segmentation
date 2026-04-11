"""
Data Augmentation Module
Implements spatial and intensity augmentations for BraTS data
"""

import numpy as np
from typing import Tuple, Union, List, Optional
import logging

from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter

from config import AUGMENTATION_SETTINGS

logger = logging.getLogger(__name__)


class SpatialAugmentation:
    """Spatial transformations: rotation, flip, elastic deformation"""

    @staticmethod
    def random_rotate(
        image: np.ndarray,
        angle_range: Tuple[float, float] = (-15, 15),
        axes: Tuple[int, int] = (1, 2),
    ) -> np.ndarray:
        """
        Apply random rotation around specified axes

        Args:
            image: Input array (C, D, H, W) or (D, H, W)
            angle_range: Random angle range in degrees
            axes: Axes around which to rotate

        Returns:
            Rotated image
        """
        angle = np.random.uniform(angle_range[0], angle_range[1])
        rotated = ndimage.rotate(image, angle, axes=axes, reshape=False, order=1)
        return rotated

    @staticmethod
    def random_flip(
        image: np.ndarray,
        flip_axes: List[int] = [1, 2],
        flip_probability: float = 0.5,
    ) -> np.ndarray:
        """
        Apply random flips along specified axes

        Args:
            image: Input array
            flip_axes: Axes to randomly flip
            flip_probability: Probability of flipping each axis

        Returns:
            Flipped image
        """
        flipped = image.copy()
        for axis in flip_axes:
            if np.random.rand() < flip_probability:
                flipped = np.flip(flipped, axis=axis)
        return flipped

    @staticmethod
    def elastic_deformation(
        image: np.ndarray,
        alpha: Tuple[float, float] = (30, 30),
        sigma: Tuple[float, float] = (3, 3),
    ) -> np.ndarray:
        """
        Apply elastic deformation using random displacement fields

        Args:
            image: Input array (2D or 3D)
            alpha: Range for displacement magnitude (min, max)
            sigma: Standard deviation of Gaussian for smoothing displacement

        Returns:
            Deformed image
        """
        shape = image.shape[-2:]  # Get H, W

        # Generate random displacement fields
        alpha_x = np.random.uniform(alpha[0], alpha[1])
        alpha_y = np.random.uniform(alpha[0], alpha[1])

        dx = np.random.randn(*shape) * alpha_x
        dy = np.random.randn(*shape) * alpha_y

        # Smooth displacement fields
        dx = gaussian_filter(dx, sigma=sigma[0])
        dy = gaussian_filter(dy, sigma=sigma[1])

        # Apply to each slice if 3D
        if image.ndim == 3:  # (D, H, W)
            deformed = np.zeros_like(image)
            for slice_idx in range(image.shape[0]):
                deformed[slice_idx] = SpatialAugmentation._apply_elastic_deformation_2d(
                    image[slice_idx], dx, dy
                )
        elif image.ndim == 4:  # (C, D, H, W)
            deformed = np.zeros_like(image)
            for c in range(image.shape[0]):
                for slice_idx in range(image.shape[1]):
                    deformed[c, slice_idx] = SpatialAugmentation._apply_elastic_deformation_2d(
                        image[c, slice_idx], dx, dy
                    )
        else:
            deformed = SpatialAugmentation._apply_elastic_deformation_2d(image, dx, dy)

        return deformed

    @staticmethod
    def _apply_elastic_deformation_2d(
        image_2d: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
    ) -> np.ndarray:
        """Apply elastic deformation to 2D image"""
        h, w = image_2d.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Remove distortion
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Map coordinates
        distorted = map_coordinates(
            image_2d,
            indices,
            order=1,
            cval=0.0,
            prefilter=False,
        )

        return distorted.reshape(image_2d.shape)

    @staticmethod
    def random_crop(
        image: np.ndarray,
        crop_size: Union[int, Tuple[int, int]],
    ) -> np.ndarray:
        """
        Random crop from image

        Args:
            image: Input array
            crop_size: Crop size (single int or tuple for 2D crop)

        Returns:
            Cropped image
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h, w = image.shape[-2:]
        ch, cw = crop_size

        if h <= ch and w <= cw:
            return image

        top = np.random.randint(0, h - ch + 1) if h > ch else 0
        left = np.random.randint(0, w - cw + 1) if w > cw else 0

        if image.ndim == 3:
            return image[:, top : top + ch, left : left + cw]
        elif image.ndim == 4:
            return image[:, :, top : top + ch, left : left + cw]

        return image[top : top + ch, left : left + cw]


class IntensityAugmentation:
    """Intensity transformations: brightness, contrast, gamma, noise"""

    @staticmethod
    def adjust_brightness(
        image: np.ndarray,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
    ) -> np.ndarray:
        """
        Randomly adjust brightness (multiply by factor in range)

        Args:
            image: Input array
            brightness_range: Range for brightness multiplier

        Returns:
            Brightness-adjusted image
        """
        factor = np.random.uniform(brightness_range[0], brightness_range[1])
        return image * factor

    @staticmethod
    def adjust_contrast(
        image: np.ndarray,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ) -> np.ndarray:
        """
        Adjust contrast around mean

        Args:
            image: Input array
            contrast_range: Range for contrast scaling

        Returns:
            Contrast-adjusted image
        """
        factor = np.random.uniform(contrast_range[0], contrast_range[1])
        mean = image.mean()
        adjusted = mean + factor * (image - mean)
        return adjusted

    @staticmethod
    def gamma_correction(
        image: np.ndarray,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
    ) -> np.ndarray:
        """
        Apply gamma correction (power-law transformation)
        Formula: image^(1/gamma)

        Args:
            image: Input array (should be in [0, 1] range)
            gamma_range: Range for gamma values

        Returns:
            Gamma-corrected image
        """
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])

        # Ensure values are in [0, 1]
        img_min = image.min()
        img_max = image.max()

        if img_max > img_min:
            image_norm = (image - img_min) / (img_max - img_min)
        else:
            image_norm = image

        # Apply gamma
        gamma_corrected = np.power(image_norm, 1.0 / gamma)

        # Scale back to original range
        result = gamma_corrected * (img_max - img_min) + img_min
        return result

    @staticmethod
    def add_gaussian_noise(
        image: np.ndarray,
        std: float = 0.01,
    ) -> np.ndarray:
        """
        Add Gaussian noise

        Args:
            image: Input array
            std: Standard deviation of Gaussian noise

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, std, image.shape)
        return image + noise

    @staticmethod
    def intensity_shift(
        image: np.ndarray,
        shift_range: Tuple[float, float] = (-0.2, 0.2),
    ) -> np.ndarray:
        """
        Shift intensity values

        Args:
            image: Input array
            shift_range: Range for intensity shift as fraction of mean

        Returns:
            Shifted image
        """
        shift = np.random.uniform(shift_range[0], shift_range[1])
        mean = np.abs(image.mean())
        shift_amount = shift * mean
        return image + shift_amount


class CompositeAugmentation:
    """Combines multiple augmentations"""

    def __init__(self, augmentation_settings: dict = None):
        """
        Initialize with augmentation settings

        Args:
            augmentation_settings: Dictionary with augmentation parameters
        """
        self.settings = augmentation_settings or AUGMENTATION_SETTINGS

    def augment(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        probability: float = 0.5,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Apply augmentation pipeline

        Args:
            image: Input image (C, D, H, W) or (D, H, W)
            segmentation: Optional segmentation mask (same shape as image or without C)
            probability: Probability of applying augmentation

        Returns:
            Augmented image, optionally with augmented segmentation
        """
        if np.random.rand() > probability:
            if segmentation is not None:
                return image, segmentation
            return image

        # Spatial augmentations
        if self.settings.get("rotate_range"):
            image = SpatialAugmentation.random_rotate(
                image,
                angle_range=self.settings["rotate_range"],
            )
            if segmentation is not None:
                segmentation = SpatialAugmentation.random_rotate(
                    segmentation,
                    angle_range=self.settings["rotate_range"],
                )

        if self.settings.get("horizontal_flip"):
            image = SpatialAugmentation.random_flip(image, flip_axes=[2])
            if segmentation is not None:
                segmentation = SpatialAugmentation.random_flip(
                    segmentation, flip_axes=[2]
                )

        if self.settings.get("vertical_flip"):
            image = SpatialAugmentation.random_flip(image, flip_axes=[1])
            if segmentation is not None:
                segmentation = SpatialAugmentation.random_flip(
                    segmentation, flip_axes=[1]
                )

        if self.settings.get("elastic_deformation"):
            image = SpatialAugmentation.elastic_deformation(
                image,
                alpha=self.settings["elastic_alpha"],
                sigma=self.settings["elastic_sigma"],
            )
            if segmentation is not None:
                segmentation = SpatialAugmentation.elastic_deformation(
                    segmentation,
                    alpha=self.settings["elastic_alpha"],
                    sigma=self.settings["elastic_sigma"],
                )

        # Intensity augmentations
        if self.settings.get("intensity_shifts"):
            image = IntensityAugmentation.intensity_shift(
                image,
                shift_range=self.settings["intensity_shift_range"],
            )

        if self.settings.get("brightness_range"):
            image = IntensityAugmentation.adjust_brightness(
                image,
                brightness_range=self.settings["brightness_range"],
            )

        if self.settings.get("contrast_range"):
            image = IntensityAugmentation.adjust_contrast(
                image,
                contrast_range=self.settings["contrast_range"],
            )

        if self.settings.get("gamma_range"):
            image = IntensityAugmentation.gamma_correction(
                image,
                gamma_range=self.settings["gamma_range"],
            )

        if self.settings.get("noise_std"):
            image = IntensityAugmentation.add_gaussian_noise(
                image,
                std=self.settings["noise_std"],
            )

        if segmentation is not None:
            return image, segmentation

        return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test augmentation
    test_image = np.random.randn(4, 155, 240, 240)
    test_seg = np.random.randint(0, 4, (155, 240, 240))

    augmentor = CompositeAugmentation()

    # Test individual augmentations
    print("Testing Spatial Augmentations...")
    rotated = SpatialAugmentation.random_rotate(test_image[:, 0])
    print(f"Rotation: {test_image[:, 0].shape} -> {rotated.shape}")

    flipped = SpatialAugmentation.random_flip(test_image[:, 0])
    print(f"Flip: {test_image[:, 0].shape} -> {flipped.shape}")

    # Test composite augmentation
    print("\nTesting Composite Augmentation...")
    augmented_img, augmented_seg = augmentor.augment(
        test_image, test_seg, probability=1.0
    )
    print(f"Augmented image: {augmented_img.shape}")
    print(f"Augmented segmentation: {augmented_seg.shape}")
    print(f"Image range: [{augmented_img.min():.2f}, {augmented_img.max():.2f}]")
