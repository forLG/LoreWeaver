"""
Graph utilities for node/edge deduplication and merging.

Uses unified schema from utils/graph_schemas.py.
"""
from typing import Any

from src.qwen3_8b.utils.graph_schemas import Edge, KnowledgeGraph, Node


def deduplicate_graph(graph: dict[str, Any] | KnowledgeGraph) -> KnowledgeGraph:
    """
    Remove duplicate nodes and edges from a graph.

    Args:
        graph: Graph dict with 'nodes' and 'edges' keys, or KnowledgeGraph instance

    Returns:
        Deduplicated KnowledgeGraph
    """
    if isinstance(graph, KnowledgeGraph):
        nodes = graph.nodes
        edges = graph.edges
    else:
        nodes_dict = graph.get("nodes", [])
        edges_dict = graph.get("edges", [])
        nodes = [Node(**n) for n in nodes_dict]
        edges = [Edge(**e) for e in edges_dict]

    # Deduplicate nodes by ID (keep the one with more complete data)
    unique_nodes: dict[str, Node] = {}
    for node in nodes:
        if node.id not in unique_nodes:
            unique_nodes[node.id] = node
        else:
            existing = unique_nodes[node.id]
            # Keep the version with more complete label
            if len(node.label) > len(existing.label):
                unique_nodes[node.id] = node

    # Deduplicate edges
    unique_edges: list[Edge] = []
    seen_edges: set[str] = set()
    for edge in edges:
        edge_key = f"{edge.source}|{edge.relation}|{edge.target}"
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_edges.append(edge)

    return KnowledgeGraph(nodes=list(unique_nodes.values()), edges=unique_edges)


def merge_graphs(sub_graphs: list[dict[str, Any] | KnowledgeGraph]) -> KnowledgeGraph:
    """
    Merge multiple graphs into one, deduplicating nodes and edges.

    Args:
        sub_graphs: List of graph dicts or KnowledgeGraph instances

    Returns:
        Merged KnowledgeGraph
    """
    all_nodes: list[Node] = []
    all_edges: list[Edge] = []

    for g in sub_graphs:
        if isinstance(g, KnowledgeGraph):
            all_nodes.extend(g.nodes)
            all_edges.extend(g.edges)
        else:
            all_nodes.extend([Node(**n) for n in g.get("nodes", [])])
            all_edges.extend([Edge(**e) for e in g.get("edges", [])])

    return deduplicate_graph(KnowledgeGraph(nodes=all_nodes, edges=all_edges))


def apply_entity_mapping(
    graph: dict[str, Any] | KnowledgeGraph,
    mapping: dict[str, str]
) -> KnowledgeGraph:
    """
    Apply entity ID mapping to a graph (for deduplication).

    Args:
        graph: Graph dict or KnowledgeGraph with nodes and edges
        mapping: Dict mapping old IDs to canonical IDs

    Returns:
        KnowledgeGraph with remapped node IDs and edge endpoints
    """
    if isinstance(graph, KnowledgeGraph):
        nodes = graph.nodes
        edges = graph.edges
    else:
        nodes = [Node(**n) for n in graph.get("nodes", [])]
        edges = [Edge(**e) for e in graph.get("edges", [])]

    final_nodes: dict[str, Node] = {}
    final_edges: list[Edge] = []
    seen_edges: set[str] = set()

    # Remap nodes
    for node in nodes:
        canonical_id = mapping.get(node.id, node.id)

        # Create new node with updated ID
        node_data = node.model_dump()
        node_data["id"] = canonical_id

        if canonical_id not in final_nodes:
            final_nodes[canonical_id] = Node(**node_data)
        else:
            # Keep the version with more complete label
            if len(node.label) > len(final_nodes[canonical_id].label):
                final_nodes[canonical_id] = Node(**node_data)

    # Remap edges
    for edge in edges:
        source = mapping.get(edge.source, edge.source)
        target = mapping.get(edge.target, edge.target)

        # Skip self-loops
        if source == target:
            continue

        edge_key = f"{source}|{edge.relation}|{target}"

        if edge_key not in seen_edges:
            new_edge = Edge(
                source=source,
                target=target,
                relation=edge.relation
            )
            final_edges.append(new_edge)
            seen_edges.add(edge_key)

    return KnowledgeGraph(nodes=list(final_nodes.values()), edges=final_edges)


def format_graph_summary(graph: dict[str, Any] | KnowledgeGraph) -> str:
    """Format a graph as a readable summary string."""
    if isinstance(graph, KnowledgeGraph):
        nodes = graph.nodes
        edges = graph.edges
    else:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

    nodes_text = "\n".join([
        f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Unknown')})"
        for n in nodes
    ])
    edges_text = "\n".join([
        f"- {e.get('source')} -> {e.get('target')} ({e.get('relation', 'related_to')})"
        for e in edges
    ])
    return f"NODES:\n{nodes_text}\n\nEDGES:\n{edges_text}"
