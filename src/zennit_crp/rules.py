import functools
import weakref

from zennit.core import Hook, RemovableHandle, RemovableHandleList


class MaskHook(Hook):
    """Mask hooks for adaptive gradient masking or simple modification."""

    def __init__(self, masks=None):
        super().__init__()
        if masks is None:
            masks = [self._default_mask]
        self.masks = masks
    
    def pre_backward(self, module, grad_input, grad_output):
        """Hook applied during backward-pass"""
        masked_grad_output = grad_output[0].clone()
        for mask in self.masks:
            masked_grad_output = mask(masked_grad_output)
        super().pre_backward(module, grad_input, (masked_grad_output,))
        return (masked_grad_output,)

    def copy(self):
        """Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same masks.
        """
        return MaskHook(masks=self.masks)

    def remove(self):
        """When removing hooks, remove all stored mask_fn."""
        self.masks.clear()
        super().remove()

    @staticmethod
    def _default_mask(obj):
        return obj


class FeatVisHook:
    """Feature Visualization hooks for reference sampling inside forward and backward passes."""

    def __init__(self, FV, concept, layer_name, dict_inputs, on_device):
        """
        Parameters:
            dict_inputs: contains sample_indices and targets inputs to FV.analyze_activation and FV.analyze_relevance
        """

        self.FV = FV
        self.concept = concept
        self.layer_name = layer_name
        self.dict_inputs = dict_inputs
        self.on_device = on_device

    def post_forward(self, module, input, output):
        """Register a backward-hook to the resulting tensor right after the forward."""

        s_indices, targets = (
            self.dict_inputs["sample_indices"],
            self.dict_inputs["targets"],
        )
        activation = (
            output.detach().to(self.on_device) if self.on_device else output.detach()
        )
        self.FV.analyze_activation(
            activation, self.layer_name, self.concept, s_indices, targets
        )

        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad):
            return hook_ref().backward(module, grad)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            # only if gradient required
            output[0].register_hook(wrapper)
        return output[0] if len(output) == 1 else output

    def backward(self, module, grad):
        """Hook applied during backward-pass"""

        s_indices, targets = (
            self.dict_inputs["sample_indices"],
            self.dict_inputs["targets"],
        )
        relevance = (
            grad.detach().to(self.on_device) if self.on_device else grad.detach()
        )
        self.FV.analyze_relevance(
            relevance, self.layer_name, self.concept, s_indices, targets
        )

        return grad

    def copy(self):
        """Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same stored_grads list.
        """
        return self.__class__(
            self.FV, self.concept, self.layer_name, self.dict_inputs, self.on_device
        )

    def remove(self):
        pass

    def register(self, module):
        """Register this instance by registering the neccessary forward hook to the supplied module."""
        return RemovableHandleList(
            [
                RemovableHandle(self),
                module.register_forward_hook(self.post_forward),
            ]
        )
