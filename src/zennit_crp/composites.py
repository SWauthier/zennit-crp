"""Composites, registered in a global composite dict."""

import torch
from zennit.composites import NameMapComposite, register_composite
from zennit.core import Composite

from zennit_crp.rules import Mask


@register_composite("condition")
class Condition(NameMapComposite):
    """Composite that applies a mask to the gradients at specified layers.
    Each condition corresponds to a batch.

    Parameters
    ----------
    conditions: list of dict
        Each dictionary in the list contains module names as keys and lists of concept indices as values.
    mask_map: callable or dict, optional
        If callable, it should be a function that takes (batch_id, concept_ids) as arguments and returns a function
        that modifies the gradient. If dict, it should map module names to such functions.
        If None, a default mask function is used that zeros out all but the specified concept indices.
    name_map: list of (list of str, Hook), optional
        A mapping as a list of tuples, with a tuple of applicable module names and a Hook. This will be prepended to
        the ``name_map`` defined by the composite.
    canonizers: list of :py:class:`zennit.canonizers.Canonizer`, optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(self, conditions, mask_map=None, name_map=None, canonizers=None):
        if mask_map is None:
            mask_map = self._default_mask
        if name_map is None:
            name_map = []

        hook_map = {}
        masked_modules = set()
        for i, condition in enumerate(conditions):
            for module_name, concept_ids in condition.items():
                if module_name not in hook_map:
                    hook_map[module_name] = Mask([])
                mask_fn = self._mask_fn(mask_map, i, concept_ids, module_name)
                hook_map[module_name].masks.append(mask_fn)
                masked_modules.add(module_name)

        name_map = name_map + [([name], hook) for name, hook in hook_map.items()]

        super().__init__(name_map=name_map, canonizers=canonizers)
        self.masked_modules = masked_modules

    @staticmethod
    def _mask_fn(mask_map, batch_id, concept_ids, module_name):
        if callable(mask_map):
            return mask_map(batch_id, concept_ids)
        elif isinstance(mask_map, dict):
            return mask_map[module_name](batch_id, concept_ids)
        else:
            raise ValueError("<mask_map> must be a dictionary or callable function.")

    @staticmethod
    def _default_mask(batch_id: int, concept_ids: list):
        """
        Wrapper that generates a function that modifies the gradient.

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            integer lists corresponding to channel indices.
        layer_name: str, optional
            Name of the layer where the mask is applied.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fn(grad):
            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            masked_tensor = grad.clone()
            masked_tensor[batch_id] = mask * masked_tensor[batch_id]
            return masked_tensor

        return mask_fn


class MultiComposite(Composite):
    """Composite that applies multiple composites sequentially.

    Since this implementation of LRP uses hooks during backward pass,
    composites registered last will be applied first during the backward pass.

    Parameters
    ----------
    composites: `list[Composite]`
        A list of Composites. The list order of composites defines their matching order.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(self, composites, canonizers=None):
        if canonizers is None:
            canonizers = []
        self.composites = composites
        super().__init__(
            module_map=self.mapping,
            canonizers=sum(
                [composite.canonizers for composite in composites], canonizers
            ),
        )

    def mapping(self, ctx, name, module):
        """Get the appropriate hook given a list of composites.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found by the first match in the composite list,
            or None if no applicable hook was found.
        """
        # create a context for each of the sub-composites if ctx is empty
        if not ctx:
            ctx.update({composite: {} for composite in self.composites})

        # create list of hooks by evaluating module maps of all given composites
        hooks = [
            composite.module_map(ctx[composite], name, module)
            for composite in self.composites
        ]

        # return all hooks that are not None
        return (hook for hook in hooks if hook is not None)

    def register(self, module):
        """Apply all canonizers and register all hooks to a module (and its recursive children).
        Previous canonizers of this composite are reverted and all hooks registered by this composite are removed.
        The module or any of its children (recursively) may still have other hooks attached.

        Parameters
        ----------
        module: :py:class:`torch.nn.Module`
            Hooks and canonizers will be applied to this module recursively according to ``module_map`` and
            ``canonizers``.
        """
        self.remove()

        for canonizer in self.canonizers:
            self.handles += canonizer.apply(module)

        ctx = {}
        for name, child in module.named_modules():
            templates = self.module_map(ctx, name, child)
            for template in templates:
                hook = template.copy()
                self.hook_refs.add(hook)
                self.handles.append(hook.register(child))
