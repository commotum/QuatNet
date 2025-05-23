# isokawa_pytorch_layer.py
import torch
import torch.nn as nn
import quatnet_cuda # This will be the name from setup.py, e.g., 'isokawa_cuda_ops'

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
        if not x_batch.is_cuda: x_batch = x_batch.cuda()
        if not W_q.is_cuda: W_q = W_q.cuda()
        if not theta_q.is_cuda: theta_q = theta_q.cuda()

        # Ensure inputs are contiguous
        x_batch_c = x_batch.contiguous()
        W_q_c = W_q.contiguous()
        theta_q_c = theta_q.contiguous()

        M_in = x_batch_c.shape[1]
        N_out = W_q_c.shape[0]

        # Call the C++ backend
        # isokawa_forward_cpp returns a pair: (y_out_tensor, s_internal_tensor)
        output_Y, s_internal = quatnet_cuda.isokawa_forward_cpp(
            x_batch_c, W_q_c, theta_q_c, M_in, N_out
        )

        # Save tensors needed for backward pass
        # Crucially, save S_internal (pre-activation)
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
        if not grad_output_Y.is_cuda: grad_output_Y = grad_output_Y.cuda()
        grad_output_Y_c = grad_output_Y.contiguous()

        x_batch, W_q, theta_q, s_internal = ctx.saved_tensors
        M_in = ctx.M_in
        N_out = ctx.N_out

        # Call the C++ backward function stub
        # This will compute dL/dX, dL/dW, dL/dTheta
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

        # Xavier/Kaiming-like initialization for quaternion components
        # For W_q, each of the 4 components could be N(0, std^2)
        # Effective fan_in for each component might be considered input_dim
        # Effective fan_out for each component might be output_dim
        stdv_w = (2.0 / (input_dim + output_dim))**0.5
        # Each component is a random variable. If they are i.i.d. with var sigma^2,
        # then norm of quaternion is related to Chi-distribution.
        # A simpler approach: initialize components from a distribution with this stdv.
        
        self.W_q = nn.Parameter(torch.empty(output_dim, input_dim, 4))
        nn.init.normal_(self.W_q, mean=0.0, std=stdv_w / 2.0) # Divide std by 2 as 4 components contribute to variance

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
        # It's good practice to ensure the input x_batch is purely imaginary if the layer expects it.
        # This can be done in the data loader or here.
        # if x_batch.requires_grad: # Only do this if x_batch itself is not a parameter
        #    x_batch_pure = x_batch.clone()
        #    x_batch_pure[:, :, 0] = 0.0
        # else:
        #    x_batch[:, :, 0] = 0.0 # Modify in-place if it's safe
        #    x_batch_pure = x_batch
        # For safety, data loader should prepare pure quaternions.

        return IsokawaLayerGPUFunction.apply(x_batch, self.W_q, self.theta_q)

    def __repr__(self):
        return f"IsokawaPytorchLayer(input_dim={self.input_dim}, output_dim={self.output_dim})"