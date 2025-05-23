# test_import.py
import torch
from isokawa_pytorch_layer import IsokawaPytorchLayer # Make sure this points to your file

def run_test():
    # Automatically select CUDA if available, else CPU (though your layer needs CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        # Fallback or error if CUDA is essential and not found
        print("CUDA is NOT available. This test requires CUDA.")
        try:
            device = torch.device("cuda:0") 
            torch.cuda.current_device() 
        except Exception as e:
            print(f"Could not initialize CUDA device: {e}")
            print("Please ensure CUDA is available and PyTorch can access it.")
            return

    print(f"Using device: {device}")

    batch_size = 2
    input_dim_q = 16 # Number of input pure quaternions
    output_dim_q = 4 # Number of output pure quaternions

    # Create layer and move to device
    try:
        layer = IsokawaPytorchLayer(input_dim=input_dim_q, output_dim=output_dim_q).to(device)
        print(f"Layer created: {layer}")
        print(f"Layer weights shape: {layer.W_q.shape}, device: {layer.W_q.device}")
        print(f"Layer thresholds shape: {layer.theta_q.shape}, device: {layer.theta_q.device}")

        x_numpy = torch.rand(batch_size, input_dim_q, 4, dtype=torch.float32)
        x_numpy[:, :, 0] = 0.0 # Set w component to 0 for pure quaternions
        x_tensor = x_numpy.to(device)
        
        print(f"Input tensor shape: {x_tensor.shape}, device: {x_tensor.device}, dtype: {x_tensor.dtype}")

        # Forward pass
        output_tensor = layer(x_tensor) 
        print(f"Output tensor shape: {output_tensor.shape}, device: {output_tensor.device}, dtype: {output_tensor.dtype}")
        print("Output (first sample, first quaternion):", output_tensor[0, 0, :].tolist())
        print("Forward pass successful.")

        if output_tensor.requires_grad:
            print("Attempting backward pass (with stubbed C++ backward)...")
            # Calculate dummy loss using the x, y, z components (indices 1, 2, 3)
            # output_tensor[:, :, 0] should be 0 for pure quaternion outputs
            imaginary_components = output_tensor[:, :, 1:] # Slices to get x, y, z
            dummy_loss = imaginary_components.sum()
            
            dummy_loss.backward()
            print("Backward pass stub executed.")
            
            if layer.W_q.grad is not None:
                print(f"Gradient for W_q exists. Shape: {layer.W_q.grad.shape}, Sum of abs: {layer.W_q.grad.abs().sum().item()}")
            else:
                print("W_q.grad is None. Check autograd setup or if W_q requires_grad.")
            
            if layer.theta_q.grad is not None:
                print(f"Gradient for theta_q exists. Shape: {layer.theta_q.grad.shape}, Sum of abs: {layer.theta_q.grad.abs().sum().item()}")
            else:
                print("theta_q.grad is None. Check autograd setup or if theta_q requires_grad.")
        else:
            print("Output tensor does not require grad. Check layer parameters and autograd.Function.")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()