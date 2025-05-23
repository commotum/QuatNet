# python/quatnet_pytorch_layer.py
import torch
import torch.nn as nn
import quatnet_cuda # Your compiled C++ module

# Mapping string activation names to enum values expected by C++
activation_map = {
    "NONE": quatnet_cuda.ActivationType.NONE,
    "SPLIT_TANH": quatnet_cuda.ActivationType.SPLIT_TANH,
    "SPLIT_SIGMOID": quatnet_cuda.ActivationType.SPLIT_SIGMOID,
    "PURE_IMAG_SIGMOID": quatnet_cuda.ActivationType.PURE_IMAG_SIGMOID,
}

class QuatNetDenseLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_input, W, B, layer_id_capsule):
        # x_input: (batch_size, M_in_q, 4)
        # W: (N_out_q, M_in_q, 4) -> These are PyTorch nn.Parameters
        # B: (N_out_q, 4)         -> These are PyTorch nn.Parameters
        # layer_id_capsule: A PyCapsule holding the int64_t layer_id for the C++ object
        
        # This design is tricky: PyTorch manages W and B. The C++ layer also has W and B.
        # They need to be synchronized. For forward, we'd copy PyTorch W, B to C++ layer's W, B.
        # This is inefficient.
        # A better design: C++ functions take W and B tensor pointers directly.
        # For now, let's assume the C++ layer object (identified by layer_id)
        # has its W and B already set/initialized, and this forward call uses them.
        # And the W, B passed here are just for autograd to know they are parameters.

        layer_id = layer_id_capsule.get_pointer() # Assuming capsule stores a pointer to int64_t or just int64_t

        # Ensure tensors are on the correct device and contiguous
        x_input_c = x_input.contiguous().cuda()
        
        # Call the C++ forward function which uses the C++ layer's internal W and B
        output_Y = quatnet_cuda.dense_forward(layer_id, x_input_c)

        # For backward, we need x_input. The C++ layer should have saved it.
        # We also need W for dL/dX. The C++ layer has its W.
        # We need pre-activations S for d(Activation)/dS. The C++ layer should have saved it.
        ctx.save_for_backward(x_input_c) # Only save input, W & B are part of the layer object
        ctx.layer_id_capsule = layer_id_capsule # Keep layer_id for backward

        return output_Y

    @staticmethod
    def backward(ctx, grad_output_Y):
        x_input_saved, = ctx.saved_tensors
        layer_id = ctx.layer_id_capsule.get_pointer()

        grad_output_Y_c = grad_output_Y.contiguous().cuda()

        # Call C++ backward. It calculates dL/dX and accumulates dL/dW, dL/dB internally.
        grad_X = quatnet_cuda.dense_backward(layer_id, grad_output_Y_c, x_input_saved)
        
        # Retrieve dL/dW and dL/dB from the C++ layer
        # These need to be exposed via bindings if not already.
        # For now, assume they are handled by optimizer calling update_weights on C++ obj.
        # Autograd needs gradients for W and B returned here.
        
        # This requires functions like:
        # grad_W_tensor = quatnet_cuda.get_weight_gradient_cpp(layer_id)
        # grad_B_tensor = quatnet_cuda.get_bias_gradient_cpp(layer_id)
        # These should return copies of the C++ layer's accumulated gradients.
        
        # Placeholder for actual retrieval of gradients for W and B
        # If W and B were direct inputs to forward, grad_W and grad_B would be calculated by C++ backward
        # and returned here.
        # Since W and B are managed by the C++ object, this autograd.Function doesn't directly return their grads.
        # This breaks the standard autograd if W and B are nn.Parameters.

        # Correct approach if W, B are nn.Parameters:
        # 1. C++ dense_forward and dense_backward must take W and B tensors.
        # 2. dense_backward must return (grad_X, grad_W, grad_B).
        # The current binding for dense_backward only returns grad_X.
        
        # Given the current C++ bindings (which are evolving), this is a simplification:
        # We'd need to modify bindings to return grad_W and grad_B.
        # For now, returning None for W and B grads, assuming optimizer step is on C++ side.
        # This is NOT how you'd typically integrate with PyTorch optimizers.
        
        # Let's assume the bindings `dense_backward` is changed to match Parcollet:
        # grad_X, grad_W, grad_B = quatnet_cuda.parcollet_dense_backward_cpp(...)
        # For now, this class cannot properly provide gradients for PyTorch nn.Parameters W and B.
        
        # Returning None for W and B gradients to PyTorch, as the C++ layer manages them.
        # This means PyTorch optimizers won't update W and B directly.
        # The update must be called on the C++ object.
        # This is a non-standard PyTorch pattern.

        grad_W_placeholder = torch.zeros_like(ctx.layer_id_capsule.W_ref_for_shape) # Need a way to get shape
        grad_B_placeholder = torch.zeros_like(ctx.layer_id_capsule.B_ref_for_shape) # Need a way to get shape

        return grad_X, grad_W_placeholder, grad_B_placeholder, None # For W, B, layer_id_capsule


class QuatNetPytorchDenseLayer(nn.Module):
    def __init__(self, input_dim_q, output_dim_q, activation_str="NONE"):
        super().__init__()
        self.input_dim_q = input_dim_q
        self.output_dim_q = output_dim_q
        
        activation_type_enum_val = activation_map.get(activation_str.upper())
        if activation_type_enum_val is None:
            raise ValueError(f"Unknown activation_str: {activation_str}. Valid are {list(activation_map.keys())}")

        # Create and store the C++ layer object instance ID
        # This layer_id will be passed to the autograd.Function
        self.layer_id = quatnet_cuda.create_dense_layer(input_dim_q, output_dim_q, activation_type_enum_val)
        
        # Create PyTorch nn.Parameters. These will be the "source of truth"
        # and need to be synced with the C++ layer's internal weights/biases.
        # This synchronization is the tricky part with stateful C++ objects.
        # A more standard approach makes C++ functions stateless, taking W and B from PyTorch.

        # For now, this nn.Module will NOT hold its own W and B parameters if the C++ side manages them.
        # If C++ manages them, how does PyTorch optimizer work?
        # This indicates the C++ layer's update_weights would be called.
        
        # Let's assume the C++ layer is initialized with random weights.
        # PyTorch nn.Module doesn't need its own W, B if C++ handles everything.
        # But for optim.Adam(model.parameters()), it needs nn.Parameter.

        # This current structure is more like a C++-driven layer.
        # To make it fully PyTorch-idiomatic for nn.Parameters:
        # 1. IsokawaLayerGPUFunction.forward takes W and B as PyTorch tensors.
        # 2. IsokawaLayerGPUFunction.backward computes and returns dL/dW and dL/dB.
        # 3. The C++ bindings (isokawa_forward_cpp, isokawa_backward_cpp) must accept W, B tensor data.
        # My C++ bindings from the *previous* turn (for the Isokawa *rotational* layer)
        # were designed this way (taking W, B tensors).
        # The current bindings.cpp for QuaternionDenseLayer uses a layer_id.

        # To keep it simple with the current bindings.cpp that uses layer_id:
        # We make dummy parameters so PyTorch optimizer has something to work on,
        # but the actual forward/backward/update happens via the C++ object.
        # This is NOT ideal but shows one way to bridge.
        self._dummy_param = nn.Parameter(torch.empty(0)) # To make model.parameters() non-empty

        # For gradcheck, we need to pass actual W and B.
        # Let's assume we modify the C++ bindings and autograd.Function to accept W and B.
        # The Isokawa autograd function was correctly structured to accept W, B.
        # We need to do the same for a Parcollet-style dense layer.
        
        # Sticking to the Parcollet style, and aiming for PyTorch idiomatic nn.Parameters:
        stdv_w = (2.0 / (input_dim_q * 4 + output_dim_q * 4))**0.5 
        self.W = nn.Parameter(torch.empty(output_dim_q, input_dim_q, 4))
        nn.init.normal_(self.W, mean=0.0, std=stdv_w)

        self.B = nn.Parameter(torch.empty(output_dim_q, 4))
        nn.init.normal_(self.B, mean=0.0, std=0.01)
        
        self.activation_type_enum_val = activation_type_enum_val


    def forward(self, x_batch):
        # This would call an autograd function that takes self.W and self.B
        # For this to work, your C++ bindings `dense_forward` and `dense_backward`
        # need to be adapted to take W and B tensor pointers, not a layer_id.
        # This means modifying src/bindings.cpp and the C++ layer's forward/backward methods
        # OR creating new C-style C++ functions that perform the layer logic using given W, B.
        
        # Placeholder: This requires a `ParcolletDenseLayerFunction(torch.autograd.Function)`
        # that calls C++ functions like `parcollet_dense_forward_cpp(x, W, B, act_type)`
        # and `parcollet_dense_backward_cpp(...)` which returns (grad_X, grad_W, grad_B).
        
        # For now, this will fail as QuatNetDenseFunction is not defined this way.
        # The user needs to decide on the C++/Python interface for parameters.
        raise NotImplementedError("This layer needs a proper autograd.Function that takes W and B tensors "
                                  "and calls C++ functions designed for that.")
        # return QuatNetDenseFunction.apply(x_batch, self.W, self.B, self.layer_id_capsule_or_act_type)


    def __del__(self):
        if hasattr(self, 'layer_id'):
             quatnet_cuda.destroy_dense_layer(self.layer_id)

    def __repr__(self):
        return (f"QuatNetPytorchDenseLayer(input_dim_q={self.input_dim_q}, output_dim_q={self.output_dim_q})")