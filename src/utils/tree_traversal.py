"""
Tree traversal utilities for processing shadow tree structures.
"""
from collections.abc import Callable
from typing import Any


def collect_sections(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Recursively collect all section nodes from a tree.

    Args:
        nodes: List of node dicts with potential 'children' keys

    Returns:
        Flat list of all section nodes
    """
    collected = []
    for node in nodes:
        collected.append(node)
        if "children" in node:
            collected.extend(collect_sections(node["children"]))
    return collected


def collect_sections_with_context(
    nodes: list[dict[str, Any]],
    parent_title: str = "",
    parent_id: str = ""
) -> list[dict[str, Any]]:
    """
    Recursively collect sections with parent context information.

    Args:
        nodes: List of node dicts
        parent_title: Title of parent node (for recursion)
        parent_id: ID of parent node (for recursion)

    Returns:
        List of context dicts with keys: id, title, parent_title, content, child_titles
    """
    collected = []
    for node in nodes:
        current_title = node.get("title", "Untitled")
        current_id = node.get("id", "")

        children = node.get("children", [])
        child_titles = [child.get("title", "Untitled") for child in children]

        context_obj = {
            "id": current_id,
            "title": current_title,
            "parent_title": parent_title,
            "parent_id": parent_id,
            "content": node.get("content", ""),
            "child_titles": child_titles
        }

        # Only collect nodes with actual content
        if context_obj["content"]:
            collected.append(context_obj)

        if children:
            collected.extend(collect_sections_with_context(
                children, current_title, current_id
            ))

    return collected


def filter_sections_by_content(
    sections: list[dict[str, Any]],
    min_length: int = 0
) -> list[dict[str, Any]]:
    """
    Filter sections to only those with meaningful content.

    Args:
        sections: List of section dicts
        min_length: Minimum content length to include

    Returns:
        Filtered list of sections
    """
    return [
        s for s in sections
        if s.get("content") and len(s.get("content", "")) >= min_length
    ]


def traverse_tree(
    nodes: list[dict[str, Any]],
    visitor: Callable[[dict[str, Any]], None],
    post_order: bool = False
) -> None:
    """
    Generic tree traversal with a visitor function.

    Args:
        nodes: List of node dicts
        visitor: Function to call on each node (mutates node in place)
        post_order: If True, visit children before parent (post-order)
    """
    for node in nodes:
        if "children" in node:
            if post_order:
                traverse_tree(node["children"], visitor, post_order=True)
                visitor(node)
            else:
                visitor(node)
                traverse_tree(node["children"], visitor, post_order=False)
        else:
            visitor(node)


def find_nodes_by_type(
    nodes: list[dict[str, Any]],
    node_type: str
) -> list[dict[str, Any]]:
    """
    Find all nodes of a specific type.

    Args:
        nodes: List of node dicts
        node_type: Type value to match

    Returns:
        List of matching nodes
    """
    results = []

    def _collector(node: dict[str, Any]) -> None:
        if node.get("type") == node_type:
            results.append(node)

    traverse_tree(nodes, _collector)
    return results


def find_node_by_id(
    nodes: list[dict[str, Any]],
    node_id: str
) -> dict[str, Any] | None:
    """
    Find a node by its ID.

    Args:
        nodes: List of node dicts
        node_id: ID to search for

    Returns:
        Matching node dict or None
    """
    for node in nodes:
        if node.get("id") == node_id:
            return node
        if "children" in node:
            result = find_node_by_id(node["children"], node_id)
            if result:
                return result
    return None


def count_nodes(nodes: list[dict[str, Any]], node_type: str | None = None) -> int:
    """
    Count nodes in tree, optionally filtered by type.

    Args:
        nodes: List of node dicts
        node_type: Optional type filter

    Returns:
        Count of matching nodes
    """
    count = 0

    def _counter(node: dict[str, Any]) -> None:
        nonlocal count
        if node_type is None or node.get("type") == node_type:
            count += 1

    traverse_tree(nodes, _counter)
    return count
