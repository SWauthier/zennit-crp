"""Hooks for gradient masking, layer recording, and feature visualization.

Hooks are the core mechanism for modifying the backward pass in zennit.
This module provides CRP-specific hooks that extend zennit's Hook base class.
"""

from __future__ import annotations

import torch
from zennit.core import Hook


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


class RecordingHook(Hook):
    """Hook that records layer outputs and their gradients.

    Extends zennit's :py:class:`~zennit.core.Hook` to store the output tensor
    during the forward pass and call :py:meth:`torch.Tensor.retain_grad` so
    that the gradient (relevance) is available after the backward pass.

    Uses Hook's standard registration (:py:meth:`~zennit.core.Hook.register`),
    returning a :py:class:`~zennit.core.RemovableHandleList` and integrating
    properly with composites.

    Attributes
    ----------
    activation : torch.Tensor or None
        The detached activation recorded during the forward pass.
    relevance : torch.Tensor or None
        The detached gradient (relevance) available after backward.
    """

    def __init__(self):
        super().__init__()
        self.activation: torch.Tensor | None = None
        self.relevance: torch.Tensor | None = None

    @property
    def output(self):
        """The recorded output tensor, or ``None`` if not yet recorded."""
        return self.stored_tensors.get("output")

    def forward(self, module, args, kwargs, output):
        """Store the output tensor and enable gradient retention.

        Parameters
        ----------
        module : torch.nn.Module
            The module to which this hook is attached.
        args : tuple[torch.Tensor]
            The input tensors passed to ``module.forward``.
        kwargs : dict
            The keyword arguments passed to ``module.forward``.
        output : torch.Tensor
            The output tensor.
        """
        self.stored_tensors["output"] = output
        output.retain_grad()

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
        output = self.output
        if output is None:
            return

        # Activation: detached copy of the forward output
        act = output.detach()[:length]
        self.activation = act.to(on_device) if on_device else act

        # Relevance: gradient at this layer (populated by retain_grad)
        if output.grad is not None:
            rel = output.grad.detach()[:length]
            self.relevance = rel.to(on_device) if on_device else rel
            output.grad = None
        else:
            self.relevance = torch.zeros_like(self.activation)

    def reset(self):
        """Clear stored tensors for reuse across batches."""
        self.activation = None
        self.relevance = None
        self.stored_tensors.pop("output", None)

    def copy(self):
        """Return a copy of this hook.

        Returns
        -------
        RecordingHook
            A fresh recording hook.
        """
        return RecordingHook()

    def remove(self):
        """Remove the registered hooks and clear stored tensors."""
        self.reset()
        super().remove()


class FeatVisHook(Hook):
    """Hook for reference sampling during feature visualization.

    Extends zennit's :py:class:`~zennit.core.Hook` to record activations
    during the forward pass and analyze relevances during the backward pass,
    passing both to a :py:class:`~zennit_crp.visualization.FeatureVisualization`
    instance.

    Uses Hook's :py:meth:`~zennit.core.Hook.forward` for activation recording
    and :py:meth:`~zennit.core.Hook.pre_backward` for relevance analysis,
    replacing the manual tensor-level hook that was previously required.

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
        super().__init__()
        self.fv = fv
        self.concept = concept
        self.layer_name = layer_name
        self.context = context
        self.on_device = on_device

    def forward(self, module, args, kwargs, output):
        """Record activations after the forward pass.

        Parameters
        ----------
        module : torch.nn.Module
            The module to which this hook is attached.
        args : tuple[torch.Tensor]
            The input tensors.
        kwargs : dict
            The keyword arguments.
        output : torch.Tensor
            The output tensor.
        """
        sample_indices = self.context["sample_indices"]
        targets = self.context["targets"]

        activation = output.detach()
        if self.on_device:
            activation = activation.to(self.on_device)

        self.fv.analyze_activation(activation, self.layer_name, self.concept, sample_indices, targets)

    def pre_backward(self, module, grad_input, grad_output):
        """Analyze relevance during the backward pass.

        Parameters
        ----------
        module : torch.nn.Module
            The module to which this hook is attached.
        grad_input : tuple[torch.Tensor]
            The input gradient tensors.
        grad_output : tuple[torch.Tensor]
            The output gradient tensors.
        """
        super().pre_backward(module, grad_input, grad_output)

        sample_indices = self.context["sample_indices"]
        targets = self.context["targets"]

        relevance = grad_output[0].detach()
        if self.on_device:
            relevance = relevance.to(self.on_device)

        self.fv.analyze_relevance(relevance, self.layer_name, self.concept, sample_indices, targets)

    def copy(self):
        """Return a copy of this hook sharing the same feature visualization instance.

        Returns
        -------
        FeatVisHook
            A copy of this hook.
        """
        return FeatVisHook(self.fv, self.concept, self.layer_name, self.context, self.on_device)
