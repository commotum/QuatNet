import torch
from isokawa_pytorch_layer import IsokawaPytorchLayer

# Create a test layer
layer = IsokawaPytorchLayer(input_dim=16, output_dim=4)

# Create a test input (batch_size=2, input_dim=16, quaternion_components=4)
x = torch.randn(2, 16, 4)

# Forward pass
output = layer(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}") 