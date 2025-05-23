# train_compression.py
import torch
import torch.nn as nn
# from isokawa_pytorch_layer import IsokawaPytorchLayer # Old
from quatnet_pytorch_layer import QuatNetPytorchDenseLayer # New name for clarity
from data_loader import create_data_loaders
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompressionAutoencoder(nn.Module):
    def __init__(self, input_dim=16*1, hidden_dim=4*1, patch_pixels=16): # input_dim is num_quaternions
        super().__init__()
        self.patch_pixels = patch_pixels # Should be input_dim

        # Using QuatNetPytorchDenseLayer (Parcollet-style WX+B, if correctly bound)
        # For this example, it's still wired to call the Isokawa rotational layer's autograd function.
        # To truly use Parcollet, QuatNetPytorchDenseLayer.forward would call a different autograd.Function
        # that itself calls C++ bindings for your original QuaternionDenseLayer.
        
        # Let's assume input_dim is the number of quaternions (e.g., 16 for a 4x4 patch)
        # Hidden layer activation: SPLIT_TANH (operates on all 4 components)
        # Output layer activation: PURE_IMAG_SIGMOID (operates on x,y,z; w=0)
        # These activation types need to be passed to and handled by your C++ layer / autograd.Function

        # This setup implies QuatNetPytorchDenseLayer needs an activation_type argument
        # that its corresponding autograd.Function and C++ backend can use.
        # The IsokawaLayerGPUFunction currently hardcodes its activation logic.
        # This is a simplification for now.
        
        print(f"Autoencoder: Input Quaternions: {input_dim}, Hidden Quaternions: {hidden_dim}")
        self.encoder = QuatNetPytorchDenseLayer(input_dim, hidden_dim) # Should ideally take activation_type
        self.decoder = QuatNetPytorchDenseLayer(hidden_dim, input_dim) # Should ideally take activation_type
                                                                    # and the output layer needs PURE_IMAG_SIGMOID logic.

        # For image compression, the Isokawa paper's approach (pure quaternion signals, specific activation)
        # means the current IsokawaLayerGPUFunction is more aligned, despite its complex W gradient.
        # If using a Parcollet-style layer, the output activation needs careful handling.

    def forward(self, x):
        # x shape: (batch_size, num_quaternions_per_patch, 4)
        # Ensure input x is purely imaginary for the Isokawa model
        # (data_loader should already do this)
        # x[:,:,0] = 0.0 # Enforce if necessary

        encoded_q = self.encoder(x) # Output might be full quaternion if encoder uses SPLIT_TANH
        
        # If encoder output (hidden) is full quaternion, but decoder expects pure:
        # This depends on the design. Isokawa implies pure signals between layers.
        # If your Parcollet-style QuaternionDenseLayer always processes full Q, that's fine.
        # The IsokawaLayerGPUFunction we built assumes pure inputs if theta is pure and X is pure.
        # Let's assume for Isokawa, hidden is also pure.
        # This implies the encoder's activation should also result in pure quaternions if following Isokawa strictly.
        # Our current IsokawaLayerGPUFunction's activation *does* produce pure quaternions.

        reconstructed_q = self.decoder(encoded_q) # Decoder output must be pure for RGB
        return reconstructed_q

def train_model(model, train_loader, test_loader, num_epochs=10, batch_size=32,
                     learning_rate=0.001, device='cuda', save_dir='models_quatnet'):
    logger.info(f"Starting training on {device}...")
    model.to(device)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Will apply to all 4 components if not careful

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()

        for batch_idx, patches_batch in enumerate(train_loader):
            # Data loader provides (num_images_in_batch, num_patches_per_image, pixels_in_patch, 4)
            # We need to reshape to (total_patches, pixels_in_patch, 4)
            # Or, if DataLoader gives one image's patches at a time: (num_patches, pixels_in_patch, 4)
            # Current DataLoader gives (1, num_patches, pixels_in_patch, 4) if batch_size=1 for DataLoader
            # Let's assume patches_batch is already (current_batch_of_patches, pixels_per_patch, 4)
            
            # If DataLoader batch_size > 1, then patches_batch could be a list of tensors, or needs careful collation.
            # For simplicity, assume DataLoader batch_size refers to number of *images*, and we process patches.
            # Let's adjust create_data_loaders to yield batches of patches directly.

            # Assuming patches_batch from DataLoader is (batch_of_patches, patch_pixels, 4)
            data_q = patches_batch.to(device) # This is X, target is also X for autoencoder

            # Ensure w-component of input data is zero (pure quaternion for Isokawa model)
            # This should ideally be handled by the data loader more robustly.
            data_q_pure = data_q.clone() # Avoid modifying original data if it's used elsewhere
            data_q_pure[:, :, 0] = 0.0

            optimizer.zero_grad()
            output_q = model(data_q_pure)

            # Loss on imaginary components only
            # Output's real part should be 0 from PURE_IMAG_SIGMOID, target's real part is 0
            loss = criterion(output_q[:,:,1:], data_q_pure[:,:,1:]) # Compare x,y,z parts
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            if batch_idx % 50 == 0: # Log every 50 batches
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for patches_batch in test_loader:
                data_q = patches_batch.to(device)
                data_q_pure = data_q.clone()
                data_q_pure[:, :, 0] = 0.0
                
                output_q = model(data_q_pure)
                loss = criterion(output_q[:,:,1:], data_q_pure[:,:,1:])
                epoch_val_loss += loss.item()
        
        avg_epoch_val_loss = epoch_val_loss / len(test_loader)
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} Summary: Train Loss: {avg_epoch_train_loss:.6f}, Val Loss: {avg_epoch_val_loss:.6f}, Time: {epoch_time:.2f}s")

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(model.state_dict(), save_path / "best_compression_model.pth")
            logger.info(f"Saved new best model to {save_path / 'best_compression_model.pth'}")
        
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), save_path / f"checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    # Configuration
    PATCH_SIZE = 4 # Isokawa used 4x4 patches
    NUM_INPUT_QUATERNIONS = PATCH_SIZE * PATCH_SIZE # 16
    NUM_HIDDEN_QUATERNIONS = 4 # Isokawa used 4

    # Modify create_data_loaders if it doesn't already return batches of patches
    # For now, assume DataLoader batch_size is for number of images, and we iterate patches
    # It's better if DataLoader directly yields (batch_of_patches, patch_pixels, 4)
    
    # Data loader should collate patches from multiple images into a single batch of patches
    # This requires a custom collate_fn or a dataset that returns individual patches.
    # Let's assume `create_data_loaders` is adapted for this.
    # If `QuaternionImageDataset.__getitem__` returns all patches from one image (N_patches, patch_dim, 4),
    # then DataLoader with batch_size=1 will give one such tensor. We need to iterate or reshape.
    # A simpler DataLoader setup for this script:
    # Dataset returns one patch (patch_dim, 4), DataLoader collates them.
    # This needs `QuaternionImageDataset` to be modified.

    logger.info("Starting Isokawa-style Image Compression Training with Parcollet-inspired Layers")
    logger.info(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE} -> {NUM_INPUT_QUATERNIONS} input quaternions")
    logger.info(f"Hidden layer: {NUM_HIDDEN_QUATERNIONS} quaternions")

    # For this script to run, QuaternionImageDataset's __getitem__ should return a *single* patch
    # and DataLoader's collate_fn will stack them into a batch.
    # Or, modify the training loop to handle list of patch tensors.
    # Simplest change for now: assume create_data_loaders gives batches of patches.
    
    # Adjust batch_size for create_data_loaders to mean "batch of patches"
    # The current data_loader.py has batch_size for DataLoader, which means number of images.
    # Let's make that clear or adjust it. For now, assume it's a batch of patches.
    # If one image of 256x256 with 4x4 patches -> (256/4)^2 = 64^2 = 4096 patches.
    # A DataLoader batch_size of 1 for images is too slow if we iterate patches inside.
    
    # The create_data_loaders in data_loader.py is likely set up for batches of images,
    # where each item from the dataset is (num_patches_in_image, patch_pixels, 4).
    # We need to adjust the training loop or data loader for patch-level batching.
    # For now, this script assumes create_data_loaders yields batches of patches directly.
    # This would require `QuaternionImageDataset` to return individual patches and `DataLoader` to batch them.

    model = CompressionAutoencoder(input_dim=NUM_INPUT_QUATERNIONS, hidden_dim=NUM_HIDDEN_QUATERNIONS)
    
    # Dummy data for shape checking if dataloader is complex
    # dummy_batch_of_patches = torch.rand(32, NUM_INPUT_QUATERNIONS, 4) # (batch, 16, 4)
    # dummy_batch_of_patches[:,:,0] = 0
    # print("Dummy input shape:", dummy_batch_of_patches.shape)
    # with torch.no_grad():
    #    dummy_output = model(dummy_batch_of_patches.cuda())
    # print("Dummy output shape:", dummy_output.shape)


    train_model(model,
                train_dir='data/train',
                test_dir='data/test',
                num_epochs=50, # Reduced for quick test
                batch_size=256, # Number of patches per batch
                learning_rate=0.001,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )