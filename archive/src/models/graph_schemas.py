"""
Pydantic models for heterogeneous knowledge graph validation.

Provides type-safe schemas for:
- Entity nodes (static knowledge)
- Event nodes (dynamic narrative)
- Edge relations
- Full graph structure
"""
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EntityNode(BaseModel):
    """Entity node in the knowledge graph (static knowledge)."""

    id: str = Field(..., description="Unique entity identifier (snake_case)")
    label: str = Field(..., description="Display name of the entity")
    type: str = Field(..., description="Entity type: Location, Creature, Item, Group")
    node_type: Literal["entity"] = Field(default="entity", description="Node type discriminator")
    aliases: list[str] = Field(default_factory=list, description="Alternative names/references")
    source_node: str | None = Field(None, description="Source section where entity was found")

    # Semantic properties for enhanced entity typing
    location_type: str | None = Field(
        None,
        description="For Location entities: area, building, room, feature, or path"
    )
    creature_type: str | None = Field(
        None,
        description="For Creature entities: humanoid, beast, undead, construct, etc."
    )
    is_generic: bool = Field(
        default=False,
        description="True if entity is a generic group (e.g., 'zombies', 'kobolds') rather than named"
    )
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional entity properties")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID is snake_case."""
        if not v:
            raise ValueError("Entity ID cannot be empty")
        # Basic snake_case check: lowercase, underscores, numbers
        if not all(c.islower() or c == "_" or c.isdigit() for c in v):
            raise ValueError(f"Entity ID must be snake_case, got: {v}")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate entity type."""
        valid_types = {"Location", "Creature", "Item", "Group"}
        if v not in valid_types:
            raise ValueError(f"Invalid entity type: {v}. Must be one of {valid_types}")
        return v


class EventNode(BaseModel):
    """Event node in the knowledge graph (dynamic narrative)."""

    id: str = Field(..., description="Unique event identifier (snake_case)")
    label: str = Field(..., description="Event name/title")
    type: str = Field(..., description="Event type: encounter, combat, discovery, dialogue, exploration, observation, quest, trap, puzzle")
    node_type: Literal["event"] = Field(default="event", description="Node type discriminator")
    participants: list[str] = Field(default_factory=list, description="Entity IDs participating in this event")
    location: str | None = Field(None, description="Location entity ID where event occurs")
    description: str = Field(default="", description="Description of what happened")
    source_node: str | None = Field(None, description="Source section where event was found")

    # Temporal and causal semantics
    triggered_by: str | None = Field(None, description="Event ID, condition, or trigger that starts this event")
    outcomes: list[str] = Field(default_factory=list, description="Event IDs that result from this event")
    conditions: dict[str, Any] = Field(default_factory=dict, description="Prerequisites (e.g., {\"has_key\": true})")
    sequence: int | None = Field(None, description="Order within location (1, 2, 3...)")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional event properties")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID is snake_case."""
        if not v:
            raise ValueError("Event ID cannot be empty")
        if not all(c.islower() or c == "_" or c.isdigit() for c in v):
            raise ValueError(f"Event ID must be snake_case, got: {v}")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate event type."""
        valid_types = {
            "encounter", "combat", "discovery", "dialogue", "exploration", "observation",
            "quest", "trap", "puzzle", "social_challenge"
        }
        if v not in valid_types:
            raise ValueError(f"Invalid event type: {v}. Must be one of {valid_types}")
        return v


class Edge(BaseModel):
    """Edge/relation between nodes in the knowledge graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation: str = Field(..., description="Relation type: has_participant, occurs_at, inhabits, etc.")
    description: str | None = Field(None, description="Human-readable description of the relation")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional edge properties")

    @field_validator("source", "target")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID is not empty."""
        if not v:
            raise ValueError("Node ID cannot be empty")
        return v

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, v: str) -> str:
        """Validate relation type."""
        if not v:
            raise ValueError("Relation type cannot be empty")
        # Domain-specific relation types (extend as needed)
        common_relations = {
            # Event relations
            "has_participant", "occurs_at", "triggers", "caused_by", "prevents",
            # Spatial relations
            "part_of", "connected_to", "leads_to",
            # Creature-state relations
            "inhabits", "guards", "patrols", "hidden_at", "trapped_at",
            # Social/Political relations
            "commands", "serves", "allied_with", "rival_of", "worships",
            # Item relations
            "stored_in", "locks", "unlocks", "wields", "wears", "carries",
        }
        if v not in common_relations:
            # Allow custom relations but log warning could be added
            pass
        return v


class KnowledgeGraph(BaseModel):
    """Heterogeneous knowledge graph with entity and event nodes."""

    nodes: list[EntityNode | EventNode] = Field(default_factory=list, description="All nodes in the graph")
    edges: list[Edge] = Field(default_factory=list, description="All edges in the graph")

    def get_entity_nodes(self) -> list[EntityNode]:
        """Get only entity nodes."""
        return [n for n in self.nodes if isinstance(n, EntityNode)]

    def get_event_nodes(self) -> list[EventNode]:
        """Get only event nodes."""
        return [n for n in self.nodes if isinstance(n, EventNode)]

    def get_node_by_id(self, node_id: str) -> EntityNode | EventNode | None:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node."""
        return [e for e in self.edges if e.source == node_id or e.target == node_id]

    def to_entity_only_graph(self) -> "KnowledgeGraph":
        """Return a filtered graph with only entity nodes and entity-entity edges."""
        entity_ids = {n.id for n in self.get_entity_nodes()}
        filtered_nodes = self.get_entity_nodes()
        filtered_edges = [
            e for e in self.edges
            if e.source in entity_ids and e.target in entity_ids
        ]
        return KnowledgeGraph(nodes=filtered_nodes, edges=filtered_edges)

    def model_dump_dict(self) -> dict[str, Any]:
        """Convert to plain dict format for JSON serialization."""
        return {
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges]
        }

    @classmethod
    def model_validate_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Validate and create from plain dict format."""
        # Separate nodes by type
        entity_nodes = []
        event_nodes = []

        for node_data in data.get("nodes", []):
            node_type = node_data.get("node_type")
            if node_type == "event":
                event_nodes.append(EventNode(**node_data))
            else:
                entity_nodes.append(EntityNode(**node_data))

        # Validate edges
        edges = [Edge(**e) for e in data.get("edges", [])]

        # Combine all nodes
        all_nodes: list[EntityNode | EventNode] = entity_nodes + event_nodes

        return cls(nodes=all_nodes, edges=edges)


# Helper function to create graph from raw extraction
def create_validated_graph(
    entities: list[dict],
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

    Raises:
        ValidationError: If data doesn't match schema
    """
    nodes: list[EntityNode | EventNode] = []

    # Add entity nodes
    for entity_data in entities:
        nodes.append(EntityNode(**entity_data))

    # Add event nodes if provided
    if events:
        for event_data in events:
            nodes.append(EventNode(**event_data))

    # Validate edges if provided
    validated_edges = [Edge(**e) for e in edges] if edges else []

    return KnowledgeGraph(nodes=nodes, edges=validated_edges)
