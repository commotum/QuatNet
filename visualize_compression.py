import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from train_compression import CompressionAutoencoder
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_image(image_path: str, model: CompressionAutoencoder, device: torch.device, patch_size: int = 4):
    """Load an image, process it through the model, and return both original and decompressed versions."""
    # Load and preprocess image
    logger.info(f"Loading image from {image_path}")
    img = Image.open(image_path)
    img_array = np.array(img)
    h, w, _ = img_array.shape
    logger.info(f"Image shape: {img_array.shape}")
    
    # Normalize RGB values to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Convert to quaternion patches
    s = patch_size
    rgb = img_array.reshape(h // s, s, w // s, s, 3).transpose(0, 2, 1, 3, 4)
    rgb = rgb.reshape(-1, s * s, 3)
    zeros = np.zeros(rgb.shape[:-1] + (1,), dtype=rgb.dtype)
    quat_patches = np.concatenate([zeros, rgb], axis=-1)
    logger.info(f"Quaternion patches shape: {quat_patches.shape}")
    
    # Convert to tensor and process through model
    quat_tensor = torch.FloatTensor(quat_patches).to(device)
    quat_tensor = quat_tensor.view(-1, quat_tensor.size(-2), 4)
    quat_tensor[:, :, 0] = 0.0  # ensure purely imaginary
    
    logger.info("Processing through model...")
    with torch.no_grad():
        decompressed = model(quat_tensor)
    logger.info("Model processing complete")
    
    # Convert decompressed output back to image
    decompressed = decompressed.cpu().numpy()
    decompressed_rgb = decompressed[:, :, 1:]  # Take only RGB components
    decompressed_rgb = decompressed_rgb.reshape(h // s, w // s, s, s, 3)
    decompressed_rgb = decompressed_rgb.transpose(0, 2, 1, 3, 4)
    decompressed_img = decompressed_rgb.reshape(h, w, 3)
    
    # Clip values to [0, 1] and convert to uint8
    decompressed_img = np.clip(decompressed_img, 0, 1)
    decompressed_img = (decompressed_img * 255).astype(np.uint8)
    logger.info(f"Decompressed image shape: {decompressed_img.shape}")
    
    return img_array, decompressed_img

def visualize_comparison(original: np.ndarray, decompressed: np.ndarray, save_path: str):
    """Display original and decompressed images side by side and save to file."""
    logger.info("Creating visualization...")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(decompressed)
    plt.title('Decompressed Image')
    plt.axis('off')
    
    logger.info(f"Saving comparison to {save_path}")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    logger.info("Visualization saved successfully")

def main():
    parser = argparse.ArgumentParser(description="Visualize image compression results")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model-path", type=str, default="models/best.pth", help="Path to the model checkpoint")
    parser.add_argument("--save-path", type=str, default="comparison.png", help="Path to save the comparison image")
    parser.add_argument("--patch-size", type=int, default=4, help="Size of image patches")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model with correct dimensions
    input_dim = args.patch_size * args.patch_size  # 16 for patch_size=4
    hidden_dim = 4  # Matches training configuration
    logger.info(f"Loading model from {args.model_path}")
    model = CompressionAutoencoder(input_dim, hidden_dim)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Process image
    original, decompressed = load_and_process_image(args.image_path, model, device, args.patch_size)
    
    # Visualize results
    visualize_comparison(original, decompressed, args.save_path)

if __name__ == "__main__":
    main() 