"""
Entity Resolver - Stage 3 of Knowledge Graph Pipeline

Resolves external entity references to their full data from source files.

Input:
- external_refs from adventure_parser: [(entity_type, identifier, source_id, raw, display_name)]

Output:
- resolved_entities: Full entity data with stats, lore, properties
- unmatched_entities: Entities that couldn't be found in source files
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logger import logger


@dataclass
class ResolvedEntity:
    """A fully resolved entity with data from its source file."""
    id: str                      # snake_case identifier
    label: str                   # Display name
    type: str                    # Location, Creature, Item, Group
    source_file: str             # e.g., "DoSI", "PHB"
    raw_data: dict[str, Any]     # Original JSON data
    mentions: list[str]          # List of section IDs where mentioned
    properties: dict[str, Any]   # Extracted key properties


class EntityResolver:
    """
    Resolve external entity references to their full source data.

    Looks up creatures, items, spells in their respective JSON files
    and extracts relevant properties for the knowledge graph.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

        # Cache for loaded source data
        self._creature_cache: dict[str, dict] = {}
        self._item_cache: dict[str, dict] = {}
        self._spell_cache: dict[str, dict] = {}

        # Entity name variations for fuzzy matching
        self._creature_name_index: dict[str, str] = {}  # name -> key

    def resolve(
        self,
        parsed_adventure: dict | str,
        resolve_creatures: bool = True,
        resolve_items: bool = True,
        resolve_spells: bool = True
    ) -> dict:
        """
        Resolve all external entity references.

        Args:
            parsed_adventure: Output from adventure_parser (dict or path to JSON)
            resolve_creatures: Look up creatures in bestiary files
            resolve_items: Look up items in items files
            resolve_spells: Look up spells in spell files

        Returns:
            {
                "resolved_entities": {entity_key: ResolvedEntity},
                "unmatched_entities": {entity_key: reason},
                "stats": {...}
            }
        """
        # Load parsed adventure data
        if isinstance(parsed_adventure, str):
            with open(parsed_adventure, encoding='utf-8') as f:
                adventure_data = json.load(f)
        else:
            adventure_data = parsed_adventure

        external_refs = adventure_data.get("external_refs", [])

        # Group refs by entity
        refs_by_entity = defaultdict(list)
        for ref in external_refs:
            entity_type, identifier, source_id, _raw, display_name = ref
            refs_by_entity[(entity_type, identifier, display_name)].append(source_id)

        resolved = {}
        unmatched = {}

        # Resolve each entity type
        for (entity_type, identifier, display_name), section_ids in refs_by_entity.items():
            try:
                if entity_type == "creature" and resolve_creatures:
                    entity = self._resolve_creature(identifier, display_name, section_ids)
                elif entity_type == "item" and resolve_items:
                    entity = self._resolve_item(identifier, display_name, section_ids)
                elif entity_type == "spell" and resolve_spells:
                    entity = self._resolve_spell(identifier, display_name, section_ids)
                else:
                    # Actions, backgrounds - keep as reference only
                    entity = self._create_reference_only(entity_type, identifier, display_name, section_ids)

                if entity:
                    resolved[identifier] = entity
            except Exception as e:
                # Log warning but create a default entity instead of failing
                logger.warning(f"Could not resolve {entity_type} '{display_name}' ({identifier}): {e}")
                # Create a minimal entity with default values
                entity = self._create_default_entity(entity_type, identifier, display_name, section_ids)
                resolved[identifier] = entity
                unmatched[identifier] = f"Resolution warning: {e}"

        return {
            "resolved_entities": {k: self._entity_to_dict(v) for k, v in resolved.items()},
            "unmatched_entities": unmatched,
            "stats": self._get_stats(resolved, unmatched)
        }

    # ========================================================================
    # Creature Resolution
    # ========================================================================

    def _resolve_creature(self, identifier: str, display_name: str, section_ids: list[str]) -> ResolvedEntity:
        """Resolve a creature reference from bestiary files."""
        # Parse identifier: "Name|Source" or "Name"
        parts = identifier.split("|")
        name = parts[0]
        source = parts[1] if len(parts) > 1 else None

        # Try to find in cache
        creature_data = self._find_creature(name, source)

        if not creature_data:
            raise ValueError(f"Creature not found: {name} (source: {source})")

        # Create snake_case ID
        entity_id = self._to_snake_case(name)

        # Extract key properties
        properties = {
            "size": creature_data.get("size", ["Unknown"])[0] if creature_data.get("size") else "Unknown",
            "creature_type": creature_data.get("type", "Unknown"),
            "alignment": creature_data.get("alignment", ["Unknown"])[0] if creature_data.get("alignment") else "Unknown",
            "armor_class": creature_data.get("ac", [0])[0] if creature_data.get("ac") else 0,
            "hit_points": creature_data.get("hp", {}).get("average") if creature_data.get("hp") else None,
            "speed": creature_data.get("speed", {}),
            "abilities": {
                "strength": creature_data.get("str", 10),
                "dexterity": creature_data.get("dex", 10),
                "constitution": creature_data.get("con", 10),
                "intelligence": creature_data.get("int", 10),
                "wisdom": creature_data.get("wis", 10),
                "charisma": creature_data.get("cha", 10),
            },
            "senses": creature_data.get("senses", []),
            "languages": creature_data.get("languages", []),
            "challenge_rating": creature_data.get("cr", "Unknown"),
        }

        # Add traits if present
        if "trait" in creature_data:
            properties["traits"] = [t.get("name", "") for t in creature_data["trait"]]
        if "action" in creature_data:
            properties["actions"] = [a.get("name", "") for a in creature_data["action"]]

        return ResolvedEntity(
            id=entity_id,
            label=display_name,
            type="Creature",
            source_file=creature_data.get("source", "Unknown"),
            raw_data=creature_data,
            mentions=list(set(section_ids)),
            properties=properties
        )

    def _find_creature(self, name: str, source: str | None = None) -> dict | None:
        """Find a creature in bestiary files by name."""
        # Load all bestiary files
        if not self._creature_cache:
            self._load_creatures()

        # Try exact match with source first
        if source:
            key = f"{name}|{source}"
            if key in self._creature_cache:
                return self._creature_cache[key]

        # Try exact name match
        if name in self._creature_name_index:
            return self._creature_cache[self._creature_name_index[name]]

        # Try case-insensitive match
        name_lower = name.lower()
        for cached_name, cached_key in self._creature_name_index.items():
            if cached_name.lower() == name_lower:
                return self._creature_cache[cached_key]

        # Try partial match (e.g., "Blue Dragon Wyrmling" -> "Blue Dragon Wyrmling")
        for cached_name, cached_key in self._creature_name_index.items():
            if name_lower in cached_name.lower() or cached_name.lower() in name_lower:
                return self._creature_cache[cached_key]

        return None

    def _load_creatures(self) -> None:
        """Load all creatures from bestiary JSON files."""
        bestiary_files = [
            "bestiary-dosi.json",
            "bestiary-mm.json",
        ]

        for filename in bestiary_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                continue

            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            # Handle different file structures
            monsters = data.get("monster", [])
            for monster in monsters:
                name = monster.get("name", "")
                source = monster.get("source", "")
                if not name:
                    continue

                key = f"{name}|{source}"
                self._creature_cache[key] = monster
                self._creature_name_index[name] = key

    # ========================================================================
    # Item Resolution
    # ========================================================================

    def _resolve_item(self, identifier: str, display_name: str, section_ids: list[str]) -> ResolvedEntity:
        """Resolve an item reference from items files."""
        parts = identifier.split("|")
        name = parts[0]
        source = parts[1] if len(parts) > 1 else None

        item_data = self._find_item(name, source)

        if not item_data:
            raise ValueError(f"Item not found: {name} (source: {source})")

        entity_id = self._to_snake_case(name)

        properties = {
            "rarity": item_data.get("rarity", "Unknown"),
            "type": item_data.get("type", "Unknown"),
            "requires_attunement": item_data.get("reqAttune", False),
            "wondrous": item_data.get("wondrous", False),
        }

        # Add weight if present
        if "weight" in item_data:
            properties["weight"] = item_data["weight"]

        # Add damage/weapon properties if present
        if "dmg1" in item_data:
            properties["damage"] = item_data["dmg1"]
            properties["damage_type"] = item_data.get("dmgType", "")

        # Add spell bonuses
        if "bonusSpellAttack" in item_data:
            properties["spell_attack_bonus"] = item_data["bonusSpellAttack"]
        if "bonusSpellSaveDc" in item_data:
            properties["spell_save_dc_bonus"] = item_data["bonusSpellSaveDc"]

        return ResolvedEntity(
            id=entity_id,
            label=display_name,
            type="Item",
            source_file=item_data.get("source", "Unknown"),
            raw_data=item_data,
            mentions=list(set(section_ids)),
            properties=properties
        )

    def _find_item(self, name: str, source: str | None = None) -> dict | None:
        """Find an item in items JSON files."""
        if not self._item_cache:
            self._load_items()

        # Try exact match with source
        if source:
            key = f"{name}|{source}"
            if key in self._item_cache:
                return self._item_cache[key]

        # Try name match
        name_lower = name.lower()
        for cached_name, cached_key in self._item_cache.items():
            if cached_name.lower() == name_lower:
                return self._item_cache[cached_key]

        # Try partial match
        for cached_name, cached_key in self._item_cache.items():
            if name_lower in cached_name.lower():
                return self._item_cache[cached_key]

        return None

    def _load_items(self) -> None:
        """Load all items from items JSON files."""
        item_files = ["items.json", "items-base.json"]

        for filename in item_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                continue

            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            items = data.get("item", [])
            for item in items:
                name = item.get("name", "")
                source = item.get("source", "")

                # Skip items without a name
                if not name:
                    continue

                # Ensure source is a string (handle cases where it might be dict/list)
                if not isinstance(source, str):
                    source = str(source)

                key = f"{name}|{source}"
                self._item_cache[key] = item
                self._item_cache[name] = key

    # ========================================================================
    # Spell Resolution
    # ========================================================================

    def _resolve_spell(self, identifier: str, display_name: str, section_ids: list[str]) -> ResolvedEntity:
        """Resolve a spell reference from spell files."""
        parts = identifier.split("|")
        name = parts[0]
        source = parts[1] if len(parts) > 1 else "PHB"

        spell_data = self._find_spell(name, source)

        if not spell_data:
            raise ValueError(f"Spell not found: {name} (source: {source})")

        entity_id = self._to_snake_case(name)

        properties = {
            "level": spell_data.get("level", 0),
            "school": spell_data.get("school", "Unknown"),
            "casting_time": self._parse_casting_time(spell_data.get("time", [])),
            "range": self._parse_range(spell_data.get("range", {})),
            "duration": self._parse_duration(spell_data.get("duration", [])),
            "components": spell_data.get("components", {}),
            "requires_concentration": self._requires_concentration(spell_data.get("duration", [])),
        }

        # Add damage/condition effects
        if "damageInflict" in spell_data:
            properties["damage_types"] = spell_data["damageInflict"]
        if "savingThrow" in spell_data:
            properties["saving_throw"] = spell_data["savingThrow"]

        return ResolvedEntity(
            id=entity_id,
            label=display_name,
            type="Item",  # Spells are treated as items in the schema (can be cast, found)
            source_file=spell_data.get("source", "Unknown"),
            raw_data=spell_data,
            mentions=list(set(section_ids)),
            properties=properties
        )

    def _find_spell(self, name: str, _source: str) -> dict | None:
        """Find a spell in spell JSON files."""
        if not self._spell_cache:
            self._load_spells()

        # Try exact match
        name_lower = name.lower()
        for cached_name, cached_data in self._spell_cache.items():
            if cached_name.lower() == name_lower:
                return cached_data

        return None

    def _load_spells(self) -> None:
        """Load all spells from spell JSON files."""
        spell_files = ["spells-phb.json"]

        for filename in spell_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                continue

            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            spells = data.get("spell", [])
            for spell in spells:
                name = spell.get("name", "")
                if not name:
                    continue

                self._spell_cache[name] = spell

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _create_reference_only(
        self,
        entity_type: str,
        _identifier: str,
        display_name: str,
        section_ids: list[str]
    ) -> ResolvedEntity:
        """Create a placeholder entity for things we can't resolve (actions, backgrounds)."""
        entity_id = self._to_snake_case(display_name)

        # Map to graph schema types
        type_mapping = {
            "action": "Item",  # Actions are abilities/items
            "background": "Group",  # Backgrounds are character groups
        }

        return ResolvedEntity(
            id=entity_id,
            label=display_name,
            type=type_mapping.get(entity_type, "Item"),
            source_file="Adventure",
            raw_data={},
            mentions=list(set(section_ids)),
            properties={"reference_type": entity_type}
        )

    def _create_default_entity(
        self,
        entity_type: str,
        identifier: str,
        display_name: str,
        section_ids: list[str]
    ) -> ResolvedEntity:
        """Create a default entity with minimal properties when not found in source files."""
        entity_id = self._to_snake_case(display_name)

        # Extract source from identifier if present (e.g., "Runara|DoSI")
        source = "Unknown"
        if "|" in identifier:
            parts = identifier.split("|")
            if len(parts) > 1 and parts[1]:
                source = parts[1]

        # Set default properties based on entity type
        if entity_type == "creature":
            properties = {
                "size": "Unknown",
                "creature_type": "Unknown",
                "alignment": "Unknown",
                "armor_class": 10,
                "hit_points": None,
                "speed": {},
                "abilities": {
                    "strength": 10,
                    "dexterity": 10,
                    "constitution": 10,
                    "intelligence": 10,
                    "wisdom": 10,
                    "charisma": 10,
                },
                "senses": [],
                "languages": [],
                "challenge_rating": "Unknown",
            }
        elif entity_type == "item":
            properties = {
                "rarity": "Unknown",
                "type": "Unknown",
                "requires_attunement": False,
                "wondrous": False,
            }
        elif entity_type == "spell":
            properties = {
                "level": 0,
                "school": "Unknown",
                "casting_time": "Unknown",
                "range": "Unknown",
                "duration": "Unknown",
                "components": {},
                "requires_concentration": False,
            }
        else:
            properties = {}

        return ResolvedEntity(
            id=entity_id,
            label=display_name,
            type=entity_type.capitalize(),
            source_file=source,
            raw_data={},
            mentions=list(set(section_ids)),
            properties=properties
        )

    def _to_snake_case(self, name: str) -> str:
        """Convert a name to snake_case ID."""
        # Replace spaces and special chars with underscores
        result = name.lower().replace(" ", "_").replace("-", "_").replace("'", "")
        # Remove non-alphanumeric chars except underscore
        result = "".join(c if c.isalnum() or c == "_" else "" for c in result)
        return result

    def _parse_casting_time(self, time_data: list) -> str:
        """Parse casting time from spell data."""
        if not time_data:
            return "Unknown"
        t = time_data[0]
        number = t.get("number", 1)
        unit = t.get("unit", "action")
        return f"{number} {unit}"

    def _parse_range(self, range_data: dict) -> str:
        """Parse range from spell data."""
        if not range_data:
            return "Unknown"
        range_type = range_data.get("type", "Unknown")
        distance = range_data.get("distance", {})
        if isinstance(distance, dict):
            amount = distance.get("amount", 0)
            dist_type = distance.get("type", "feet")
            return f"{range_type} {amount} {dist_type}"
        return str(distance)

    def _parse_duration(self, duration_data: list) -> str:
        """Parse duration from spell data."""
        if not duration_data:
            return "Instant"
        d = duration_data[0]
        dur_type = d.get("type", "instant")
        if dur_type == "timed":
            duration = d.get("duration", {})
            amount = duration.get("amount", 0)
            unit = duration.get("type", "round")
            return f"{amount} {unit}"
        return dur_type

    def _requires_concentration(self, duration_data: list) -> bool:
        """Check if spell requires concentration."""
        return any(d.get("concentration") for d in duration_data) if duration_data else False

    def _entity_to_dict(self, entity: ResolvedEntity) -> dict:
        """Convert ResolvedEntity to dict for JSON serialization."""
        return {
            "id": entity.id,
            "label": entity.label,
            "type": entity.type,
            "source_file": entity.source_file,
            "mentions": entity.mentions,
            "properties": entity.properties,
            # Don't include raw_data to save space
        }

    def _get_stats(self, resolved: dict, unmatched: dict) -> dict:
        """Generate statistics about resolution."""
        type_counts = defaultdict(int)
        for entity in resolved.values():
            # entity is ResolvedEntity, not dict
            type_counts[entity.type] += 1

        return {
            "total_resolved": len(resolved),
            "total_unmatched": len(unmatched),
            "by_type": dict(type_counts),
            "unmatched_list": list(unmatched.keys())
        }


# Convenience function
def resolve_entities(
    parsed_adventure: dict | str,
    data_dir: str = "data",
    output_path: str | None = None
) -> dict:
    """
    Resolve entities from parsed adventure data.

    Args:
        parsed_adventure: Output from adventure_parser (dict or path)
        data_dir: Directory containing bestiary/items/spells JSON files
        output_path: Optional path to save results

    Returns:
        Same as EntityResolver.resolve()
    """
    resolver = EntityResolver(data_dir)
    result = resolver.resolve(parsed_adventure)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    return result
