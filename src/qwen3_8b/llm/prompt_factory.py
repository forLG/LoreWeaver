"""Prompt factory for LLM extraction tasks.

Simplified to extract only core node attributes:
- id: Unique identifier
- label: Display name
- type: Entity type (location, creature, item, event, etc.)
- aliases: Alternative names
"""


class PromptFactory:
    @staticmethod
    def create_location_hierarchy_prompt_natural(
        title: str,
        content: str,
        locations_text: str
    ) -> str:
        """Extract location hierarchy (part_of) optimized for small models."""
        return f"""ROLE: Extract D&D location containment relationships.
SECTION: {title}

KNOWN LOCATIONS:
{locations_text}

TEXT:
{content}

HIERARCHY RULES:
- "X in Y" → X part_of Y
- "room in building" → room part_of building
- "feature in room" → feature part_of room
- Hierarchy: Area > Building > Room > Feature

OUTPUT:
Relation: part_of
Source: child_id
Target: parent_id
Description: context

EXAMPLES:
Relation: part_of
Source: sanctuary
Target: dragon_s_rest
Description: Inner sanctum within temple

Relation: part_of
Source: altar
Target: sanctuary
Description: Stone altar in sanctum
"""

    # ========================================================================
    # Unified Entity + Event Extraction (Heterogeneous Graph)
    # ========================================================================

    @staticmethod
    def create_unified_extraction_prompt_natural(
        title: str,
        content: str,
        parent_context: str = ""
    ) -> str:
        """
        Unified Entity + Event extraction for heterogeneous graph.
        Optimized for small models (Qwen3 8B, etc).
        """
        _nl = "\n"
        parent_line = f"PARENT: {parent_context}{_nl}" if parent_context else ""
        return f"""ROLE: Extract D&D entities and events from adventure text.
SECTION: {title}
{parent_line}
TEXT:
{content}

RULES:
1. BE AGGRESSIVE - extract ANY entity, such as creature, location, or item mentioned, no matter named or generic
2. LIST-STYLE NPC DESCRIPTIONS: When text describes multiple characters in list format ("Name does something...", "Another Name has a trait..."), extract EACH name as a separate creature entity
   - Pattern: Paragraphs starting with a proper name followed by description
   - Extract every named character, even if only briefly described
3. ONLY use <summary> for truly empty text (pure atmospheric description with no entities)

ENTITY TYPES:
- location (area/building/room/feature/path)
- creature (humanoid/beast/undead/dragon/fey/etc)
- item (objects, equipment, treasures)
- group (organizations, parties)

EVENT TYPES:
encounter, combat, discovery, dialogue, exploration, observation, quest, trap, puzzle

OUTPUT (wrap in HTML tags):
<entities>
Entity: Name
Type: location/creature/item/group
ID: snake_case
Aliases: [alias1, alias2]
</entities>

<events>
Event: Name
Type: encounter/combat/discovery/etc
Participants: [id1, id2]
Location: location_id
Description: brief description
</events>

EXAMPLES:
<entities>
Entity: Old Temple
Type: Location
ID: old_temple
Aliases: [ruin]

Entity: Bandit Guards
Type: Creature
ID: bandit_guards
Aliases: [bandits]

Entity: Thorin Ironforge
Type: Creature
ID: thorin_ironforge
Aliases: [blacksmith]

(Example: Even if "Bandit Guards" is known as a generic group, extract individual named bandits like "Grimjaw", "Silas" as separate entities)
</entities>

<events>
Event: Ambush
Type: combat
Participants: [adventurers, bandit_guards]
Location: forest_path
Description: Bandits attack the party on the road
</events>

IF NO ENTITIES/EVENTS:
<summary>
Text contains only atmospheric description with no entities.
</summary>"""

    @staticmethod
    def create_semantic_relations_prompt_natural(
        title: str,
        content: str,
        known_entities_text: str
    ) -> str:
        """
        Extract semantic relations between entities.
        """
        return f"""ROLE: Extract semantic relationships between D&D entities.
SECTION: {title}

KNOWN ENTITIES:
{known_entities_text}

TEXT:
{content}

RELATION TYPES:
- Spatial: connected_to, part_of, leads_to
- Creature: inhabits, guards, patrols, hidden_at
- Social: commands, serves, allied_with, rival_of
- Item: stored_in, locks, unlocks, wields, wears

OUTPUT:
Relation: relation_type
Source: source_id
Target: target_id
Description: context

EXAMPLE:
Relation: inhabits
Source: runara
Target: dragon_s_rest
Description: Bronze dragon lives in the temple
"""
