import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import logging

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
        h, w, c = img.shape
        patches = []
        
        # Normalize RGB values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Extract patches
        for i in range(0, h - self.patch_size + 1, self.patch_size):
            for j in range(0, w - self.patch_size + 1, self.patch_size):
                patch = img[i:i+self.patch_size, j:j+self.patch_size]
                # Convert patch to quaternion representation
                # Each pixel becomes a quaternion with w=0, x=R, y=G, z=B
                quat_patch = np.zeros((self.patch_size * self.patch_size, 4))
                for k in range(self.patch_size):
                    for l in range(self.patch_size):
                        idx = k * self.patch_size + l
                        quat_patch[idx] = [0, patch[k,l,0], patch[k,l,1], patch[k,l,2]]
                patches.append(quat_patch)
        
        return np.array(patches)

def create_data_loaders(train_dir, test_dir, batch_size=32, patch_size=8):
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader 