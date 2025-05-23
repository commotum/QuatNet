# quatnet_pytorch_layer.py
import torch
import torch.nn as nn
import quatnet_cuda # This will be the name from setup.py

class IsokawaLayerGPUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_batch, W_q, theta_q):
        """
        Forward pass for the Isokawa Quaternion Layer.

        Args:
            ctx: Context object to save tensors for backward pass.
            x_batch (torch.Tensor): Input tensor of shape (batch_size, M_in, 4).
                                    Represents pure quaternions (w-component should be 0).
            W_q (torch.Tensor): Weight parameter tensor of shape (N_out, M_in, 4).
                                Represents full quaternions.
            theta_q (torch.Tensor): Threshold parameter tensor of shape (N_out, 4).
                                    Represents pure quaternions (w-component should be 0).
        Returns:
            torch.Tensor: Output tensor Y of shape (batch_size, N_out, 4).
        """
        # Ensure tensors are on CUDA and contiguous
        # Note: Using .cuda() without checking if already on cuda is okay,
        # but can be more robust by checking x_batch.is_cuda first.
        x_batch_c = x_batch.contiguous().cuda()
        W_q_c = W_q.contiguous().cuda()
        theta_q_c = theta_q.contiguous().cuda()

        M_in = x_batch_c.shape[1]
        N_out = W_q_c.shape[0]

        # Call the C++ backend
        # isokawa_forward_cpp is expected to return a pair: (y_out_tensor, s_internal_tensor)
        output_Y, s_internal = quatnet_cuda.isokawa_forward_cpp(
            x_batch_c, W_q_c, theta_q_c, M_in, N_out
        )

        # Save tensors needed for backward pass
        # Crucially, save S_internal (pre-activation values)
        ctx.save_for_backward(x_batch_c, W_q_c, theta_q_c, s_internal)
        ctx.M_in = M_in
        ctx.N_out = N_out
        
        return output_Y

    @staticmethod
    def backward(ctx, grad_output_Y):
        """
        Backward pass for the Isokawa Quaternion Layer.

        Args:
            ctx: Context object with saved tensors.
            grad_output_Y (torch.Tensor): Gradient of the loss w.r.t. the output Y.
                                          Shape: (batch_size, N_out, 4).
        Returns:
            Tuple: Gradients w.r.t. x_batch, W_q, theta_q.
                   Shapes must match the inputs to forward.
        """
        grad_output_Y_c = grad_output_Y.contiguous().cuda()

        x_batch, W_q, theta_q, s_internal = ctx.saved_tensors
        M_in = ctx.M_in
        N_out = ctx.N_out

        # Call the C++ backward function (currently a stub for dW and dX)
        grad_X, grad_W, grad_theta = quatnet_cuda.isokawa_backward_cpp(
            grad_output_Y_c, x_batch, W_q, theta_q, s_internal,
            M_in, N_out
        )
        
        # The gradients returned must correspond to the inputs of the forward function
        # in the same order: x_batch, W_q, theta_q
        return grad_X, grad_W, grad_theta


class IsokawaPytorchLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim    # M_in: number of input pure quaternions
        self.output_dim = output_dim  # N_out: number of output pure quaternions

        # Initialize weights W_q and thresholds theta_q as nn.Parameter
        # W_q: (N_out, M_in, 4 components for full quaternion)
        # theta_q: (N_out, 4 components for pure quaternion, w=0)
        
        # Using Xavier/Kaiming-like initialization for components
        stdv_w = (2.0 / (input_dim + output_dim))**0.5 # Standard deviation
        
        self.W_q = nn.Parameter(torch.empty(output_dim, input_dim, 4))
        # Initialize each of the 4 components of W_q
        nn.init.normal_(self.W_q, mean=0.0, std=stdv_w / 2.0) # Divide std by 2 as 4 components contribute variance

        self.theta_q = nn.Parameter(torch.empty(output_dim, 4))
        nn.init.normal_(self.theta_q, mean=0.0, std=0.01) # Small initial thresholds
        with torch.no_grad():
            self.theta_q[:, 0] = 0.0 # Ensure w-component of theta is zero (pure quaternion)

    def forward(self, x_batch):
        """
        Args:
            x_batch (torch.Tensor): Input tensor of shape (batch_size, self.input_dim, 4).
                                    Represents pure quaternions (w-component should be 0).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, self.output_dim, 4).
        """
        # The IsokawaLayerGPUFunction.apply will handle moving tensors to CUDA if not already there,
        # but it's good practice to ensure model and data are on the same device beforehand.
        return IsokawaLayerGPUFunction.apply(x_batch, self.W_q, self.theta_q)

    def __repr__(self):
        # If this class is in quatnet_pytorch_layer.py but named IsokawaPytorchLayer
        return f"IsokawaPytorchLayer(input_dim={self.input_dim}, output_dim={self.output_dim})"