# test_import.py
import torch
from quatnet_pytorch_layer import IsokawaPytorchLayer, IsokawaLayerGPUFunction # Import the autograd Function too
from torch.autograd import gradcheck

def run_test(check_gradients=False): # Add a flag to enable gradcheck
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. This test requires CUDA.")
        try:
            device = torch.device("cuda:0")
            torch.cuda.current_device()
        except Exception as e:
            print(f"Could not initialize CUDA device: {e}")
            return

    print(f"Using device: {device}")

    batch_size = 2
    input_dim_q = 3 # M_in, reduced for faster gradcheck
    output_dim_q = 2 # N_out, reduced for faster gradcheck

    try:
        # For gradcheck, use double precision
        dtype_for_gradcheck = torch.double if check_gradients else torch.float32
        
        layer = IsokawaPytorchLayer(input_dim=input_dim_q, output_dim=output_dim_q).to(device).to(dtype_for_gradcheck)
        print(f"Layer created: {layer}")
        print(f"Layer W_q shape: {layer.W_q.shape}, dtype: {layer.W_q.dtype}")
        print(f"Layer theta_q shape: {layer.theta_q.shape}, dtype: {layer.theta_q.dtype}")

        x_numpy = torch.rand(batch_size, input_dim_q, 4, dtype=dtype_for_gradcheck)
        x_numpy[:, :, 0] = 0.0 
        x_tensor = x_numpy.to(device).requires_grad_(check_gradients) # Set requires_grad for gradcheck on input
        
        print(f"Input tensor shape: {x_tensor.shape}, dtype: {x_tensor.dtype}")

        if check_gradients:
            print("\n--- Running Gradcheck ---")
            # Ensure parameters also require grad for gradcheck if they are among inputs to .apply
            layer.W_q.requires_grad_(True)
            layer.theta_q.requires_grad_(True)

            # Inputs to IsokawaLayerGPUFunction.apply
            inputs_for_gradcheck = (x_tensor, layer.W_q, layer.theta_q)
            
            # gradcheck needs non-zero grad_outputs for some internal checks if outputs are not scalar
            # It's often easier to check by ensuring the function output for gradcheck is scalar,
            # or by providing a specific grad_outputs.
            # For non-scalar output, provide `grad_outputs` or set `check_undefined_grad=False` (if some inputs don't need grad)
            # Or, wrap the call in a lambda that produces a scalar:
            # test_passed = gradcheck(lambda x, w, t: IsokawaLayerGPUFunction.apply(x, w, t).sum(), inputs_for_gradcheck, eps=1e-6, atol=1e-4)
            
            # Let's test with default gradcheck settings first, it will use random grad_outputs
            # For complex functions, you might need to adjust nondet_tol or other gradcheck params
            # For gradcheck, ensure the backward pass is not just a stub for W and X, or it will likely fail for those.
            try:
                test_passed = gradcheck(IsokawaLayerGPUFunction.apply, inputs_for_gradcheck, eps=1e-6, atol=1e-3, rtol=1e-3)
                print("Gradcheck passed:", test_passed)
            except Exception as e:
                print(f"Gradcheck failed or encountered an error: {e}")
            print("--- Gradcheck Finished ---\n")


        # Regular forward pass test
        # Ensure layer is back to float if it was changed to double for gradcheck and not for regular use
        if check_gradients: # Re-cast to float for regular test if needed
            layer.to(torch.float32)
            x_tensor = x_tensor.to(torch.float32).requires_grad_(True) # Re-enable grad for float test

        output_tensor = layer(x_tensor)
        print(f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
        print("Output (first sample, first quaternion):", output_tensor[0, 0, :].tolist())
        print("Forward pass successful.")

        if output_tensor.requires_grad:
            print("Attempting backward pass...")
            imaginary_components = output_tensor[:, :, 1:]
            dummy_loss = imaginary_components.sum()
            dummy_loss.backward()
            print("Backward pass executed.")
            
            if layer.W_q.grad is not None:
                print(f"Gradient for W_q exists. Shape: {layer.W_q.grad.shape}, Sum of abs: {layer.W_q.grad.abs().sum().item()}")
            else:
                print("W_q.grad is None.")
            
            if layer.theta_q.grad is not None:
                print(f"Gradient for theta_q exists. Shape: {layer.theta_q.grad.shape}, Sum of abs: {layer.theta_q.grad.abs().sum().item()}")
            else:
                print("theta_q.grad is None.")
            if x_tensor.grad is not None:
                 print(f"Gradient for x_tensor exists. Shape: {x_tensor.grad.shape}, Sum of abs: {x_tensor.grad.abs().sum().item()}")
            else:
                print("x_tensor.grad is None.")


    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test(check_gradients=False) # Set to True after implementing more of the backward pass
    # Example: run_test(check_gradients=True)