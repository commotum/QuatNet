import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuaternionImageDataset(Dataset):
    """Dataset for loading and preprocessing images into quaternion patches."""
    
    def __init__(self, image_dir, patch_size=8):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing the images
            patch_size: Size of the patches to extract (default: 8)
        """
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.image_files = list(self.image_dir.glob('*.jpg'))
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a batch of quaternion patches from an image.
        
        Args:
            idx: Index of the image to process
            
        Returns:
            Tensor of shape (num_patches, patch_size*patch_size, 4)
            where the last dimension represents quaternion components [w,x,y,z]
        """
        # Load image
        img = Image.open(self.image_files[idx])
        img = np.array(img)
        
        # Convert to quaternion patches
        patches = self._image_to_quaternion_patches(img)
        return torch.FloatTensor(patches)
    
    def _image_to_quaternion_patches(self, img):
        """
        Convert image to quaternion patches.
        
        Args:
            img: Input image as numpy array of shape (height, width, 3)
            
        Returns:
            Array of shape (num_patches, patch_size*patch_size, 4)
            where the last dimension represents quaternion components [w,x,y,z]
        """
        h, w, _ = img.shape

        # Normalize RGB values to [0, 1]
        img = img.astype(np.float32) / 255.0

        s = self.patch_size
        # Reshape into (H/s, s, W/s, s, 3) then permute to (H/s, W/s, s, s, 3)
        rgb = img.reshape(h // s, s, w // s, s, 3).transpose(0, 2, 1, 3, 4)
        # Flatten spatial dims -> (Npatch, s*s, 3)
        rgb = rgb.reshape(-1, s * s, 3)
        zeros = np.zeros(rgb.shape[:-1] + (1,), dtype=rgb.dtype)
        # Concatenate in quaternion order [w, x, y, z] where w=0
        return np.concatenate([zeros, rgb], axis=-1)

def create_data_loaders(
    train_dir,
    test_dir,
    batch_size=32,
    patch_size=8,
    num_workers=None,
    prefetch_factor=2,
):
    """
    Create data loaders for training and testing.
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing test images
        batch_size: Batch size for the data loaders
        patch_size: Size of the patches to extract
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = QuaternionImageDataset(train_dir, patch_size)
    test_dataset = QuaternionImageDataset(test_dir, patch_size)

    if num_workers is None:
        try:
            num_workers = max(1, len(os.sched_getaffinity(0)) - 2)
        except AttributeError:
            num_workers = max(1, (os.cpu_count() or 1) - 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    
    return train_loader, test_loader 
