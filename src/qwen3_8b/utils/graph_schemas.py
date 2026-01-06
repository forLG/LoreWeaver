"""
Simplified Pydantic models for knowledge graph validation.

Core node attributes:
- id: Unique identifier
- label: Display name
- aliases: Alternative names
- type: Entity type (location, creature, item) or "event"
- extraction_method: How the node was created (llm, tag, etc.)
- internal_links_count: Count of links within same document
- external_links_count: Count of links to external documents
"""
from typing import Any

from pydantic import BaseModel, Field


class Node(BaseModel):
    """Unified node schema for both entities and events."""

    id: str = Field(..., description="Unique identifier (snake_case)")
    label: str = Field(..., description="Display name")
    type: str = Field(default="Entity", description="Type: location, creature, item, event, etc.")
    aliases: list[str] = Field(default_factory=list, description="Alternative names/references")
    extraction_method: str = Field(default="llm", description="How node was created: llm, tag, etc.")
    internal_links_count: int = Field(default=0, description="Count of internal document references")
    external_links_count: int = Field(default=0, description="Count of external document references")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class Edge(BaseModel):
    """Edge/relation between nodes in the knowledge graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation: str = Field(default="related_to", description="Relation type")
    description: str | None = Field(None, description="Human-readable description of the relation")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional edge properties")


class KnowledgeGraph(BaseModel):
    """Knowledge graph with unified nodes and edges."""

    nodes: list[Node] = Field(default_factory=list, description="All nodes in the graph")
    edges: list[Edge] = Field(default_factory=list, description="All edges in the graph")

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Get nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]

    def get_entity_nodes(self) -> list[Node]:
        """Get nodes that are not events."""
        return [n for n in self.nodes if n.type != "event"]

    def get_event_nodes(self) -> list[Node]:
        """Get event nodes."""
        return [n for n in self.nodes if n.type == "event"]

    def get_node_by_id(self, node_id: str) -> Node | None:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node."""
        return [e for e in self.edges if e.source == node_id or e.target == node_id]

    def model_dump_dict(self) -> dict[str, Any]:
        """Convert to plain dict format for JSON serialization."""
        return {
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges]
        }

    @classmethod
    def model_validate_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Validate and create from plain dict format."""
        nodes = [Node(**n) for n in data.get("nodes", [])]
        edges = [Edge(**e) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges)


# Helper function to create graph from raw extraction
def create_validated_graph(
    entities: list[dict] | None = None,
    events: list[dict] | None = None,
    edges: list[dict] | None = None
) -> KnowledgeGraph:
    """
    Create a validated KnowledgeGraph from raw extraction data.

    Args:
        entities: List of entity dicts from extraction
        events: List of event dicts from extraction (optional)
        edges: List of edge dicts from extraction (optional)

    Returns:
        Validated KnowledgeGraph instance
    """
    nodes: list[Node] = []

    # Add entity nodes
    if entities:
        for entity_data in entities:
            # Set node_type from type if not present
            if "type" not in entity_data:
                entity_data["type"] = "unknown"
            nodes.append(Node(**entity_data))

    # Add event nodes
    if events:
        for event_data in events:
            event_data["type"] = "event"
            nodes.append(Node(**event_data))

    # Validate edges
    validated_edges = [Edge(**e) for e in edges] if edges else []

    return KnowledgeGraph(nodes=nodes, edges=validated_edges)
