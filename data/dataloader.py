"""
PyTorch Dataset and DataLoader for BraTS data
Handles data loading, augmentation, and batching
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Union, Dict, Optional, List
import logging
from torch.utils.data import Dataset, DataLoader
import torch

from data.preprocessing import BraTS2021Loader, NIfTIPreprocessor
from data.augmentation import CompositeAugmentation
from config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    NUM_WORKERS,
    AUGMENTATION_PROBABILITY,
    NUM_INPUT_CHANNELS,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


class BraTS2021Dataset(Dataset):
    """
    PyTorch Dataset for BraTS 2021

    Attributes:
        Contains preprocessed MRI volumes and segmentation masks
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "Training",
        case_ids: Optional[List[str]] = None,
        augment: bool = False,
        augmentation_probability: float = AUGMENTATION_PROBABILITY,
    ):
        """
        Initialize BraTS Dataset

        Args:
            data_root: Root directory of BraTS dataset
            split: Dataset split ("Training", "Validation", "Testing")
            case_ids: Specific case IDs to use (all if None)
            augment: Whether to apply data augmentation
            augmentation_probability: Probability of applying augmentation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment
        self.augmentation_probability = augmentation_probability

        # Initialize loaders
        self.brats_loader = BraTS2021Loader(self.data_root)
        self.preprocessor = NIfTIPreprocessor()
        self.augmentor = CompositeAugmentation()

        # Get case IDs
        if case_ids is None:
            self.case_ids = self.brats_loader.get_all_case_ids(split)
        else:
            self.case_ids = case_ids

        if len(self.case_ids) == 0:
            logger.warning(
                f"No cases found in {self.data_root}/{split}. "
                "Make sure the dataset is properly downloaded."
            )

        logger.info(
            f"Loaded {len(self.case_ids)} cases from {split} split"
        )

    def __len__(self) -> int:
        """Return number of cases"""
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single case

        Args:
            idx: Index of case

        Returns:
            Dictionary with 'image' and 'segmentation' tensors
        """
        case_id = self.case_ids[idx]

        try:
            # Load case
            case_data = self.brats_loader.load_case(
                case_id,
                split=self.split,
                preprocess=True,
            )

            image = case_data["images"]  # (C, D, H, W)
            segmentation = case_data.get("segmentation", None)  # (D, H, W)

            # Apply augmentation if enabled
            if self.augment and segmentation is not None:
                image, segmentation = self.augmentor.augment(
                    image,
                    segmentation,
                    probability=self.augmentation_probability,
                )
            else:
                image = self.augmentor.augment(
                    image,
                    probability=0.0,
                )

            # Convert to tensors
            image_tensor = torch.from_numpy(image).float()

            if segmentation is not None:
                segmentation_tensor = torch.from_numpy(segmentation).long()
            else:
                segmentation_tensor = torch.zeros_like(image_tensor[0]).long()

            return {
                "image": image_tensor,
                "segmentation": segmentation_tensor,
                "case_id": case_id,
            }

        except Exception as e:
            logger.error(f"Error loading case {case_id}: {str(e)}")
            # Return zero tensors as fallback
            return {
                "image": torch.zeros((NUM_INPUT_CHANNELS, 155, 240, 240)),
                "segmentation": torch.zeros((155, 240, 240), dtype=torch.long),
                "case_id": case_id,
            }

    @staticmethod
    def create_stratified_splits(
        data_root: Union[str, Path],
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Create stratified train/val/test splits

        Args:
            data_root: Root directory of dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        loader = BraTS2021Loader(data_root)
        all_case_ids = loader.get_all_case_ids("Training")

        if len(all_case_ids) == 0:
            logger.warning("No cases found for splitting")
            return [], [], []

        # Set random seed
        np.random.seed(random_seed)
        indices = np.random.permutation(len(all_case_ids))

        # Calculate split points
        train_count = int(len(all_case_ids) * train_ratio)
        val_count = int(len(all_case_ids) * val_ratio)

        train_indices = indices[:train_count]
        val_indices = indices[train_count : train_count + val_count]
        test_indices = indices[train_count + val_count :]

        train_ids = [all_case_ids[i] for i in train_indices]
        val_ids = [all_case_ids[i] for i in val_indices]
        test_ids = [all_case_ids[i] for i in test_indices]

        logger.info(
            f"Created splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}"
        )

        return train_ids, val_ids, test_ids


class BraTS2021DataLoader:
    """Convenience wrapper for creating PyTorch DataLoaders"""

    @staticmethod
    def create_loaders(
        data_root: Union[str, Path],
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = True,
        augment_train: bool = True,
        augment_val: bool = False,
        augment_test: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders

        Args:
            data_root: Root directory of dataset
            batch_size: Batch size
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU
            augment_train: Apply augmentation to training data
            augment_val: Apply augmentation to validation data
            augment_test: Apply augmentation to test data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create splits
        train_ids, val_ids, test_ids = BraTS2021Dataset.create_stratified_splits(
            data_root
        )

        # Create datasets
        train_dataset = BraTS2021Dataset(
            data_root,
            split="Training",
            case_ids=train_ids,
            augment=augment_train,
        )

        val_dataset = BraTS2021Dataset(
            data_root,
            split="Training",
            case_ids=val_ids,
            augment=augment_val,
        )

        test_dataset = BraTS2021Dataset(
            data_root,
            split="Training",
            case_ids=test_ids,
            augment=augment_test,
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info(f"Created DataLoaders with batch_size={batch_size}")
        logger.info(
            f"Train batches: {len(train_loader)}, "
            f"Val batches: {len(val_loader)}, "
            f"Test batches: {len(test_loader)}"
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    # data_root = "data/BraTS"
    #
    # # Create a single dataset
    # dataset = BraTS2021Dataset(
    #     data_root,
    #     split="Training",
    #     augment=True,
    # )
    #
    # print(f"Dataset size: {len(dataset)}")
    # sample = dataset[0]
    # print(f"Sample keys: {sample.keys()}")
    # print(f"Image shape: {sample['image'].shape}")
    # print(f"Segmentation shape: {sample['segmentation'].shape}")
    #
    # # Create DataLoaders
    # train_loader, val_loader, test_loader = BraTS2021DataLoader.create_loaders(
    #     data_root,
    #     batch_size=8,
    #     augment_train=True,
    # )
    #
    # # Iterate through batch
    # for batch in train_loader:
    #     print(f"Batch image shape: {batch['image'].shape}")
    #     print(f"Batch segmentation shape: {batch['segmentation'].shape}")
    #     break

    print("BraTS DataLoader module loaded successfully")
