"""Hooks for gradient masking, layer recording, and feature visualization.

Hooks are the core mechanism for modifying the backward pass in zennit.
This module provides CRP-specific hooks that extend zennit's Hook base class.
"""

from __future__ import annotations

import functools
import weakref

import torch
from zennit.core import Hook, RemovableHandle, RemovableHandleList


class MaskHook(Hook):
    """Hook that applies gradient masks during the backward pass.

    Used by :py:class:`~zennit_crp.composites.MaskComposite` to implement
    conditional attribution by selectively masking relevance flows through
    specified channels.

    Parameters
    ----------
    masks : list[callable], optional
        List of mask functions to apply to the gradient. Each function has
        signature ``(grad: torch.Tensor) -> torch.Tensor``. If ``None``,
        an identity mask is used.
    """

    def __init__(self, masks: list | None = None):
        super().__init__()
        self.masks = masks if masks is not None else [self._identity]

    def pre_backward(self, module, grad_input, grad_output):
        """Apply all mask functions to the output gradient sequentially.

        Parameters
        ----------
        module : torch.nn.Module
            The module to which this hook is attached.
        grad_input : tuple[torch.Tensor]
            The input gradient tensors.
        grad_output : tuple[torch.Tensor]
            The output gradient tensors.

        Returns
        -------
        tuple[torch.Tensor]
            The masked output gradient, wrapped in a tuple.
        """
        masked = grad_output[0].clone()
        for mask in self.masks:
            masked = mask(masked)
        super().pre_backward(module, grad_input, (masked,))
        return (masked,)

    def copy(self):
        """Return a copy of this hook, sharing the same mask list.

        Returns
        -------
        MaskHook
            A copy retaining the same masks.
        """
        return MaskHook(masks=self.masks)

    def remove(self):
        """Remove all stored mask functions and clean up."""
        self.masks.clear()
        super().remove()

    @staticmethod
    def _identity(obj):
        """Identity function used as default mask."""
        return obj


class RecordingHook:
    """Forward hook that records layer outputs and their gradients.

    Stores the output tensor during forward pass and calls
    :py:meth:`torch.Tensor.retain_grad` so that the gradient (relevance)
    is available after the backward pass.

    Attributes
    ----------
    activation : torch.Tensor or None
        The detached activation recorded during the forward pass.
    relevance : torch.Tensor or None
        The detached gradient (relevance) available after backward.
    output : torch.Tensor or None
        The raw output tensor with grad_fn, usable for segmented backward.
    """

    def __init__(self):
        self.activation: torch.Tensor | None = None
        self.relevance: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _forward_hook(self, module, input, output):
        """Store the output tensor and enable gradient retention."""
        self.output = output
        output.retain_grad()

    def register(self, module: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        """Register the forward hook on a module.

        Parameters
        ----------
        module : torch.nn.Module
            The module on which to register this hook.

        Returns
        -------
        torch.utils.hooks.RemovableHandle
            Handle for removing the hook later.
        """
        self._handle = module.register_forward_hook(self._forward_hook)
        return self._handle

    def collect(self, on_device: str | torch.device | None = None, length: int | None = None):
        """Collect the recorded activation and relevance.

        Call this after the backward pass to extract the stored values.

        Parameters
        ----------
        on_device : str or torch.device, optional
            Device on which to place the collected tensors.
        length : int, optional
            Truncate tensors to this length along the batch dimension.
        """
        if self.output is None:
            return

        # Activation: detached copy of the forward output
        act = self.output.detach()[:length]
        self.activation = act.to(on_device) if on_device else act

        # Relevance: gradient at this layer (populated by retain_grad)
        if self.output.grad is not None:
            rel = self.output.grad.detach()[:length]
            self.relevance = rel.to(on_device) if on_device else rel
            self.output.grad = None
        else:
            self.relevance = torch.zeros_like(self.activation)

    def reset(self):
        """Clear stored tensors for reuse across batches."""
        self.activation = None
        self.relevance = None
        self.output = None

    def remove(self):
        """Remove the registered hook and clear stored tensors."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.reset()


class FeatVisHook:
    """Hook for reference sampling during feature visualization.

    Records activations and relevances at a specific layer and passes
    them to a :py:class:`~zennit_crp.visualization.FeatureVisualization`
    instance for analysis.

    Parameters
    ----------
    fv : FeatureVisualization
        The feature visualization instance to notify with activations/relevances.
    concept : Concept
        The concept instance used for attribution at this layer.
    layer_name : str
        Name of the layer this hook is attached to.
    context : dict
        Shared dictionary containing ``sample_indices`` and ``targets`` keys,
        updated externally before each batch.
    on_device : str or torch.device, optional
        Device on which to store intermediate results.
    """

    def __init__(self, fv, concept, layer_name: str, context: dict, on_device=None):
        self.fv = fv
        self.concept = concept
        self.layer_name = layer_name
        self.context = context
        self.on_device = on_device

    def post_forward(self, module, input, output):
        """Record activations after forward pass and register backward callback.

        Parameters
        ----------
        module : torch.nn.Module
            The module to which this hook is attached.
        input : tuple[torch.Tensor]
            The input tensors.
        output : torch.Tensor
            The output tensor.

        Returns
        -------
        torch.Tensor
            The unmodified output tensor.
        """
        sample_indices = self.context["sample_indices"]
        targets = self.context["targets"]

        activation = output.detach()
        if self.on_device:
            activation = activation.to(self.on_device)

        self.fv.analyze_activation(activation, self.layer_name, self.concept, sample_indices, targets)

        # Register a tensor-level backward hook for relevance
        hook_ref = weakref.ref(self)

        @functools.wraps(self._backward)
        def wrapper(grad):
            hook = hook_ref()
            if hook is not None:
                return hook._backward(module, grad)
            return grad

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            output[0].register_hook(wrapper)

        return output[0] if len(output) == 1 else output

    def _backward(self, module, grad):
        """Analyze relevance during backward pass."""
        sample_indices = self.context["sample_indices"]
        targets = self.context["targets"]

        relevance = grad.detach()
        if self.on_device:
            relevance = relevance.to(self.on_device)

        self.fv.analyze_relevance(relevance, self.layer_name, self.concept, sample_indices, targets)
        return grad

    def copy(self):
        """Return a copy of this hook sharing the same feature visualization instance.

        Returns
        -------
        FeatVisHook
            A copy of this hook.
        """
        return FeatVisHook(self.fv, self.concept, self.layer_name, self.context, self.on_device)

    def remove(self):
        """No-op: cleanup is handled by the handle list."""

    def register(self, module: torch.nn.Module) -> RemovableHandleList:
        """Register the forward hook on a module.

        Parameters
        ----------
        module : torch.nn.Module
            The module on which to register.

        Returns
        -------
        RemovableHandleList
            Handle list for removing the hook later.
        """
        return RemovableHandleList(
            [
                RemovableHandle(self),
                module.register_forward_hook(self.post_forward),
            ]
        )
