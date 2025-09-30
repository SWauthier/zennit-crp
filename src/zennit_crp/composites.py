from zennit.composites import NameMapComposite, register_composite

from zennit_crp.hooks import MaskHook
from zennit_crp.concepts import ChannelConcept


@register_composite("mask")
class MaskComposite(NameMapComposite):
    """Composite that applies a mask to the gradients at specified layers."""

    MODEL_OUTPUT_NAME = "y"

    def __init__(self, conditions, name_map=None, canonizers=None):
        if name_map is None:
            name_map = []

        hook_map, y_targets, modules_to_condition = {}, [], []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    mask_fn = self._mask_fn(ChannelConcept.mask, i, indices, l_name)
                    hook_map[l_name].masks.append(mask_fn)
                    if l_name not in modules_to_condition:
                        modules_to_condition.append(l_name)

        name_map = [([name], hook) for name, hook in hook_map.items()]

        super().__init__(name_map=name_map, canonizers=canonizers)

    def _mask_fn(self, mask_map, b_index, c_indices, l_name):
        if callable(mask_map):
            return mask_map(b_index, c_indices, l_name)
        elif isinstance(mask_map, dict):
            return mask_map[l_name](b_index, c_indices, l_name)
        else:
            raise ValueError("<mask_map> must be a dictionary or callable function.")
