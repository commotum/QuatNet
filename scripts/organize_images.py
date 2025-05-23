import os
import shutil
import random
from pathlib import Path

def organize_images(source_dir: str, train_dir: str, test_dir: str, test_ratio: float = 0.1):
    """
    Organize images into train and test sets with serial numbers.
    
    Args:
        source_dir: Directory containing source images
        train_dir: Directory for training images
        test_dir: Directory for test images
        test_ratio: Ratio of images to use for testing (default: 0.1)
    """
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of all image files
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * (1 - test_ratio))
    
    # Split into train and test
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]
    
    # Copy and rename files
    for idx, filename in enumerate(train_files, 1):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(train_dir, f"train_{idx:04d}.jpg")
        shutil.copy2(src, dst)
        print(f"Copied {filename} to {dst}")
    
    for idx, filename in enumerate(test_files, 1):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(test_dir, f"test_{idx:04d}.jpg")
        shutil.copy2(src, dst)
        print(f"Copied {filename} to {dst}")
    
    print(f"\nTotal images processed: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")

if __name__ == "__main__":
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "Images"
    train_dir = project_root / "data" / "train"
    test_dir = project_root / "data" / "test"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Organize images
    organize_images(str(source_dir), str(train_dir), str(test_dir)) 