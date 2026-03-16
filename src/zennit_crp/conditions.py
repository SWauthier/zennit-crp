"""Utilities for handling conditions in Concept Relevance Propagation.

Conditions map layer names to concept (channel) indices, specifying which
relevance flows to mask during the conditional backward pass.
"""

from __future__ import annotations

from collections import defaultdict

#: Name used to reference the model output in condition dictionaries.
MODEL_OUTPUT_NAME = "y"


def partition_conditions(conditions: list[dict[str, list[int]]]) -> dict[frozenset[str], list[dict[str, list[int]]]]:
    """Partition conditions into groups that share the same set of layer names.

    When ``exclude_parallel`` is enabled, conditions must be grouped by their
    layer name sets so that each group can be processed independently.

    Parameters
    ----------
    conditions : list[dict[str, list[int]]]
        List of condition dictionaries.

    Returns
    -------
    dict[frozenset[str], list[dict[str, list[int]]]]
        Mapping from frozen sets of layer names to lists of conditions
        sharing those layer names.

    Examples
    --------
    >>> conditions = [{"conv1": [0], "y": [1]}, {"conv1": [1], "y": [2]}]
    >>> partition_conditions(conditions)
    {frozenset({'conv1', 'y'}): [{'conv1': [0], 'y': [1]}, {'conv1': [1], 'y': [2]}]}
    """
    partition: dict[frozenset[str], list[dict[str, list[int]]]] = defaultdict(list)
    for condition in conditions:
        key = frozenset(condition.keys())
        partition[key].append(condition)
    return dict(partition)


def split_output_conditions(
    conditions: list[dict[str, list[int]]],
) -> tuple[list[list[int] | None], list[dict[str, list[int]]]]:
    """Separate output target conditions from layer conditions.

    Parameters
    ----------
    conditions : list[dict[str, list[int]]]
        List of condition dictionaries, potentially containing ``MODEL_OUTPUT_NAME``
        keys for output target selection.

    Returns
    -------
    tuple[list[list[int] | None], list[dict[str, list[int]]]]
        - Output targets for each condition (``None`` if no output target specified).
        - Layer-only conditions with output target entries removed.
    """
    y_targets: list[list[int] | None] = []
    layer_conditions: list[dict[str, list[int]]] = []

    for condition in conditions:
        y_targets.append(condition.get(MODEL_OUTPUT_NAME))
        layer_conditions.append({k: v for k, v in condition.items() if k != MODEL_OUTPUT_NAME})

    return y_targets, layer_conditions


def conditioned_layer_names(conditions: list[dict[str, list[int]]]) -> list[str]:
    """Extract unique layer names from conditions, excluding the output target.

    Parameters
    ----------
    conditions : list[dict[str, list[int]]]
        List of condition dictionaries.

    Returns
    -------
    list[str]
        Ordered list of unique layer names that appear in the conditions.
    """
    seen: set[str] = set()
    names: list[str] = []
    for condition in conditions:
        for name in condition:
            if name != MODEL_OUTPUT_NAME and name not in seen:
                seen.add(name)
                names.append(name)
    return names
