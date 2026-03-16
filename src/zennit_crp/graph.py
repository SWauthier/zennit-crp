"""Model graph tracing for determining layer connectivity.

Uses :py:func:`torch.jit.trace` to discover how ``nn.Module`` layers connect
to each other, enabling :py:class:`~zennit_crp.attribution.AttributionGraph`
to walk the model's computational graph.
"""

from __future__ import annotations

import torch


class GraphNode:
    """Node in the traced model graph.

    Parameters
    ----------
    node : torch.jit.Node
        A node from the JIT inlined graph.
    """

    def __init__(self, node):
        self.id = repr(node)
        self.scope_name = node.scopeName() or node.kind()

        self.output_nodes: list[GraphNode] = []
        self.input_nodes: list[GraphNode] = []
        self.is_layer = "aten" in node.kind()
        self.input_layers: list[str] = []
        self.layer_name = self.scope_name.split("__module.")[-1] if self.is_layer else ""


class ModelGraph:
    """Graph describing how layers in a PyTorch model are connected.

    Use :py:meth:`find_input_layers` to discover which layers feed into a
    given layer.

    Parameters
    ----------
    input_nodes : list
        Starting nodes of the JIT graph.
    """

    def __init__(self, input_nodes):
        self._id_node_map: dict[str, GraphNode] = {}
        self._layer_node_map: dict[str, GraphNode] = {}
        self.layer_names: list[str] = []

        for node in input_nodes:
            self._add_node(node)

    def _add_node(self, node) -> GraphNode:
        node_id = repr(node)
        if node_id not in self._id_node_map:
            graph_node = GraphNode(node)
            self._id_node_map[node_id] = graph_node
            if graph_node.is_layer:
                self._layer_node_map[graph_node.layer_name] = graph_node
        return self._id_node_map[node_id]

    def _add_connection(self, in_node, out_node) -> bool:
        node_in = self._add_node(in_node)
        node_out = self._add_node(out_node)

        new = False
        if node_in not in node_out.input_nodes:
            node_out.input_nodes.append(node_in)
            new = True
        if node_out not in node_in.output_nodes:
            node_in.output_nodes.append(node_out)
            new = True
        return new

    def set_layer_names(self, layer_names: list[str]):
        """Set the available layer names and pre-cache their input layers."""
        self.layer_names = layer_names
        for name in layer_names:
            self.find_input_layers(name)

    def find_input_layers(self, layer_name: str) -> list[str]:
        """Return layer names connected to the input of ``layer_name``.

        Only layers present in :py:attr:`layer_names` are returned.

        Parameters
        ----------
        layer_name : str
            Name of the ``nn.Module`` to query.

        Returns
        -------
        list[str]
            Input layer names.

        Raises
        ------
        KeyError
            If ``layer_name`` is not in the traced graph.
        """
        if layer_name not in self._layer_node_map:
            raise KeyError(f"Layer '{layer_name}' not found in graph.")

        root = self._layer_node_map[layer_name]
        if not root.input_layers:
            root.input_layers = self._search_inputs(root)
        return root.input_layers

    def _search_inputs(self, node: GraphNode) -> list[str]:
        found: list[str] = []
        for inp in node.input_nodes:
            if inp.is_layer and inp.layer_name in self.layer_names:
                found.append(inp.layer_name)
            else:
                found.extend(self._search_inputs(inp))
        return found

    def __str__(self) -> str:
        lines = []
        for node in self._id_node_map.values():
            targets = ", ".join(n.scope_name for n in node.output_nodes) or "end"
            lines.append(f"{node.scope_name} -> {targets}")
        return "\n".join(lines)


def trace_model_graph(
    model: torch.nn.Module,
    sample: torch.Tensor,
    layer_names: list[str],
) -> ModelGraph:
    """Trace the model and build a layer connectivity graph.

    Uses :py:func:`torch.jit.trace` to record tensor flow through the model,
    then constructs a :py:class:`ModelGraph` summarizing how named modules
    connect to each other.

    Parameters
    ----------
    model : torch.nn.Module
        The model to trace.
    sample : torch.Tensor
        An example input tensor for tracing.
    layer_names : list[str]
        Layer names to include in the graph.

    Returns
    -------
    ModelGraph
        Graph object with connectivity information.
    """
    traced = torch.jit.trace(model, (sample,), check_trace=False)
    graph = traced.inlined_graph

    node_inputs, node_outputs = _collect_node_ios(graph)
    input_nodes = _find_input_nodes(graph, node_inputs, node_outputs)

    mg = ModelGraph(input_nodes)

    for node in input_nodes:
        _build_recursive(mg, graph, node)

    mg.set_layer_names(layer_names)

    del traced, graph
    return mg


def _build_recursive(mg: ModelGraph, graph, in_node):
    """Recursively traverse the JIT graph and record connections."""
    outputs = [o.unique() for o in in_node.outputs()]
    next_nodes = _find_next_nodes(graph, outputs)

    for node in next_nodes:
        if mg._add_connection(in_node, node):
            _build_recursive(mg, graph, node)


def _find_next_nodes(graph, output_ids: list) -> list:
    """Find nodes whose inputs overlap with the given output IDs."""
    result = []
    output_set = set(output_ids)
    for node in graph.nodes():
        inputs = {i.unique() for i in node.inputs()}
        if inputs & output_set:
            result.append(node)
    return result


def _collect_node_ios(graph) -> tuple[dict, dict]:
    """Collect input/output tensor IDs per scope name."""
    inputs: dict[str, list] = {}
    outputs: dict[str, list] = {}

    for node in graph.nodes():
        if "aten" in node.kind():
            name = node.scopeName()
            if name not in inputs:
                inputs[name] = []
                outputs[name] = []
            inputs[name].extend(i.unique() for i in node.inputs())
            outputs[name].extend(o.unique() for o in node.outputs())

    return inputs, outputs


def _find_input_nodes(graph, node_inputs: dict, node_outputs: dict) -> list:
    """Find graph nodes with no incoming connections from other modules."""
    result = []
    all_outputs = {uid for ids in node_outputs.values() for uid in ids}

    for node in graph.nodes():
        if "aten" in node.kind():
            name = node.scopeName()
            if not set(node_inputs[name]) & all_outputs:
                result.append(node)

    return result
