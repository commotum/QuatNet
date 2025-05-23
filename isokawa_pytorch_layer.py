import torch
import torch.nn as nn
from quatnet_cuda import IsokawaQuaternionLayer

class IsokawaPytorchLayer(nn.Module):
    """PyTorch wrapper for the Isokawa quaternion layer."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = IsokawaQuaternionLayer(input_dim, output_dim)
        # Register weights and thresholds as nn.Parameters
        self.register_parameter('weights', nn.Parameter(torch.zeros(output_dim, input_dim, 4)))
        self.register_parameter('thresholds', nn.Parameter(torch.zeros(output_dim, 4)))
        
    def forward(self, x):
        """
        Forward pass of the Isokawa layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, 4)
               where the last dimension represents quaternion components [w,x,y,z]
        
        Returns:
            Output tensor of shape (batch_size, output_dim, 4)
        """
        # Ensure input is on the correct device
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Forward pass through the CUDA layer
        return self.layer.forward(x) 