"""
Pydantic models for data validation across the codebase.

Provides type-safe models for:
- Graph structures (nodes, edges)
- Shadow tree nodes
- Entity data
"""
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ========================================================================
# Enums
# ========================================================================

class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    LOCATION = "Location"
    ENTITY = "Entity"
    CREATURE = "Creature"
    ITEM = "Item"
    SPELL = "Spell"
    PARTY = "Party"
    WORLD = "World"
    REGION = "Region"
    ISLAND = "Island"
    CITY = "City"
    BUILDING = "Building"
    CAVE_SYSTEM = "Cave System"
    ROOM = "Room"


class RelationType(str, Enum):
    """Types of relationships between nodes."""
    CONTAINS = "contains"
    CONNECTS_TO = "connects_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    RELATED_TO = "related_to"
    ADJACENT_TO = "adjacent_to"
    NORTH_OF = "north_of"
    SOUTH_OF = "south_of"
    EAST_OF = "east_of"
    WEST_OF = "west_of"


class ShadowNodeType(str, Enum):
    """Types of nodes in the shadow tree structure."""
    SECTION = "section"
    ENTRIES = "entries"
    INSET = "inset"
    INSET_READALOUD = "insetReadaloud"


# ========================================================================
# Link Models
# ========================================================================

class Link(BaseModel):
    """Represents a parsed link from the adventure text."""
    tag: str = Field(description="Type of linked entity (creature, item, spell)")
    text: str = Field(description="Display text of the link")
    href: str | None = Field(default=None, description="Optional href URL")

    model_config = ConfigDict(frozen=True)


# ========================================================================
# Graph Models
# ========================================================================

class GraphNode(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Human-readable name")
    type: str = Field(default="Location", description="Type of the node (any string, e.g., 'Location', 'Creature', 'Item', 'Party', 'Monster', 'Container', etc.)")
    attributes: dict[str, Any] | None = Field(
        default=None,
        description="Additional attributes as key-value pairs"
    )
    name: str | None = Field(
        default=None,
        description="Alternative name field (used in some contexts)"
    )
    source: str | None = Field(
        default=None,
        description="Source reference (e.g., book name)"
    )

    model_config = ConfigDict(frozen=True)


class GraphEdge(BaseModel):
    """An edge (relationship) in the knowledge graph."""
    source: str = Field(description="ID of the source node")
    target: str = Field(description="ID of the target node")
    relation: str = Field(
        default="related_to",
        description="Type of relationship (any string, e.g., 'part_of', 'connected_to', 'inhabits', 'commands', 'stored_in', etc.)"
    )
    desc: str | None = Field(
        default=None,
        description="Human-readable description of the relationship"
    )

    model_config = ConfigDict(frozen=True)


class KnowledgeGraph(BaseModel):
    """A complete knowledge graph with nodes and edges."""
    nodes: list[GraphNode] = Field(default_factory=list, description="All nodes in the graph")
    edges: list[GraphEdge] = Field(default_factory=list, description="All edges in the graph")

    def add_node(self, node: GraphNode) -> None:
        """Add a node if not already present (by ID)."""
        existing_ids = {n.id for n in self.nodes}
        if node.id not in existing_ids:
            self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge if not already present."""
        edge_key = f"{edge.source}|{edge.relation}|{edge.target}"
        existing = {f"{e.source}|{e.relation}|{e.target}" for e in self.edges}
        if edge_key not in existing:
            self.edges.append(edge)

    def merge(self, other: "KnowledgeGraph") -> None:
        """Merge another graph into this one."""
        for node in other.nodes:
            self.add_node(node)
        for edge in other.edges:
            self.add_edge(edge)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict format for JSON serialization."""
        return {
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Create from plain dict format (JSON parsed)."""
        nodes = [GraphNode(**n) for n in data.get("nodes", [])]
        edges = [GraphEdge(**e) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges)


# ========================================================================
# Shadow Tree Models
# ========================================================================

class ShadowNode(BaseModel):
    """A node in the shadow tree structure."""
    id: str
    title: str
    type: ShadowNodeType
    content: str = Field(default="", description="Accumulated text content")
    links: list[Link] = Field(default_factory=list, description="Parsed links")
    children: list["ShadowNode"] = Field(default_factory=list, description="Child nodes")
    spatial_summary: str | None = Field(
        default=None,
        description="LLM-generated spatial summary"
    )

    def add_text(self, text: str) -> None:
        """Add text content (mutable operation)."""
        if text:
            # Note: This breaks frozen config, so we need to handle it
            object.__setattr__(self, "content", self.content + "\n" + text if self.content else text)

    def add_link(self, link: Link) -> None:
        """Add a link (mutable operation)."""
        object.__setattr__(self, "links", [*self.links, link])

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type.value,
            "content": self.content,
            "links": [link.model_dump() for link in self.links],
            "children": [c.to_dict() for c in self.children],
            **({"spatial_summary": self.spatial_summary} if self.spatial_summary else {})
        }

    # Allow mutable model for add operations
    model_config = ConfigDict(frozen=False)


# Update forward reference
ShadowNode.model_rebuild()


# ========================================================================
# Processor Config Models
# ========================================================================

class ProcessorConfig(BaseModel):
    """Configuration for LLM processors."""
    api_key: str = Field(description="OpenAI API key")
    base_url: str | None = Field(
        default="https://api.openai.com/v1",
        description="API base URL"
    )
    model: str = Field(default="gpt-4o", description="Model name")
    max_concurrent: int = Field(default=10, ge=1, le=100, description="Max concurrent requests")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int | None = Field(default=None, ge=1, description="Max tokens per response")

    model_config = ConfigDict(frozen=True)


class MultiPassConfig(ProcessorConfig):
    """Configuration for multi-pass extraction mode."""
    use_multi_pass: bool = Field(default=False, description="Enable multi-pass extraction")
    passes: list[str] = Field(
        default=["top_level", "sub_locations", "relationships", "verification"],
        description="Extraction passes to run"
    )
