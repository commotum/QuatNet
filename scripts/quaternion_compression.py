import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Quaternion:
    """Quaternion class for basic operations"""
    def __init__(self, w=0, x=0, y=0, z=0):
        self.w = w  # real part
        self.x = x  # i part
        self.y = y  # j part
        self.z = z  # k part
    
    def __mul__(self, other):
        # Quaternion multiplication
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)
    
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

class QuaternionImageDataset(Dataset):
    """Dataset for loading and preprocessing images into quaternion patches"""
    def __init__(self, image_dir, patch_size=8):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.image_files = list(self.image_dir.glob('*.jpg'))
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_files[idx])
        img = np.array(img)
        
        # Convert to quaternion patches
        patches = self._image_to_quaternion_patches(img)
        return torch.FloatTensor(patches)
    
    def _image_to_quaternion_patches(self, img):
        """Convert image to quaternion patches"""
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

class QuaternionLinear(nn.Module):
    """Quaternion linear layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights for each quaternion component
        self.weight = nn.Parameter(torch.randn(4, out_features, in_features))
        self.bias = nn.Parameter(torch.randn(4, out_features))
        
    def forward(self, x):
        # x shape: (batch_size, in_features, 4)
        # weight shape: (4, out_features, in_features)
        # bias shape: (4, out_features)
        
        # Quaternion multiplication
        w = torch.matmul(x[:,:,0], self.weight[0].t()) - \
            torch.matmul(x[:,:,1], self.weight[1].t()) - \
            torch.matmul(x[:,:,2], self.weight[2].t()) - \
            torch.matmul(x[:,:,3], self.weight[3].t()) + self.bias[0]
            
        x_out = torch.matmul(x[:,:,0], self.weight[1].t()) + \
                torch.matmul(x[:,:,1], self.weight[0].t()) + \
                torch.matmul(x[:,:,2], self.weight[3].t()) - \
                torch.matmul(x[:,:,3], self.weight[2].t()) + self.bias[1]
                
        y_out = torch.matmul(x[:,:,0], self.weight[2].t()) - \
                torch.matmul(x[:,:,1], self.weight[3].t()) + \
                torch.matmul(x[:,:,2], self.weight[0].t()) + \
                torch.matmul(x[:,:,3], self.weight[1].t()) + self.bias[2]
                
        z_out = torch.matmul(x[:,:,0], self.weight[3].t()) + \
                torch.matmul(x[:,:,1], self.weight[2].t()) - \
                torch.matmul(x[:,:,2], self.weight[1].t()) + \
                torch.matmul(x[:,:,3], self.weight[0].t()) + self.bias[3]
        
        return torch.stack([w, x_out, y_out, z_out], dim=2)

class QuaternionNetwork(nn.Module):
    """Quaternion neural network for image compression"""
    def __init__(self):
        super().__init__()
        # 64 -> 16 -> 64 architecture
        self.encoder = nn.Sequential(
            QuaternionLinear(64, 16),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            QuaternionLinear(16, 64),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, 64, 4)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, test_loader, num_epochs=100):
    """Train the quaternion network"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Print progress
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')
            
            # Evaluate on test set
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    output = model(data)
                    test_loss += criterion(output, data).item()
            logger.info(f'Test Loss: {test_loss/len(test_loader):.4f}')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create datasets
    train_dataset = QuaternionImageDataset('data/train')
    test_dataset = QuaternionImageDataset('data/test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = QuaternionNetwork()
    train_model(model, train_loader, test_loader)
    
    # Save model
    torch.save(model.state_dict(), 'models/quaternion_compression.pth')

if __name__ == '__main__':
    main() 