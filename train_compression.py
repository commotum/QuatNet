import torch
import torch.nn as nn
from isokawa_pytorch_layer import IsokawaPytorchLayer
from data_loader import create_data_loaders
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuaternionAutoencoder(nn.Module):
    """Quaternion autoencoder using Isokawa layers."""
    
    def __init__(self, input_dim=64, hidden_dim=16):
        super().__init__()
        self.encoder = IsokawaPytorchLayer(input_dim, hidden_dim)
        self.decoder = IsokawaPytorchLayer(hidden_dim, input_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_compression(train_dir, test_dir, num_epochs=100, batch_size=32, 
                     learning_rate=0.001, save_dir='models'):
    """
    Train the quaternion autoencoder.
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing test images
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        save_dir: Directory to save model checkpoints
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dir, test_dir, batch_size=batch_size
    )
    
    # Create model and move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuaternionAutoencoder().to(device)
    logger.info(f"Using device: {device}")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = model(data)
                val_loss += criterion(output, data).item()
        
        val_loss /= len(test_loader)
        
        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch}: '
                   f'Train Loss: {train_loss:.6f}, '
                   f'Val Loss: {val_loss:.6f}, '
                   f'Time: {epoch_time:.2f}s')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    train_compression(
        train_dir='data/train',
        test_dir='data/test',
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001
    ) 