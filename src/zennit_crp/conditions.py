"""Functions for handling conditions."""

from collections import defaultdict


def partition_conditions(conditions):
    """Create a partition of conditions with the same modules."""
    partition = defaultdict(list)
    for condition in conditions:
        module_names = frozenset(condition.keys())
        partition[module_names].append(condition)
    return partition
