"""
Adventure Parser - Exploration Utility

Explore parsed adventure data by type, entity, or section.
"""

import json
from collections import defaultdict
from typing import Any


class AdventureExplorer:
    """Explore parsed adventure data."""

    def __init__(self, parsed_data: dict | str):
        """
        Initialize with parsed data (dict or path to JSON file).

        Args:
            parsed_data: Either the parsed dict or path to adventure-dosi-parsed.json
        """
        if isinstance(parsed_data, str):
            with open(parsed_data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = parsed_data

        self.internal_index = self.data.get("internal_index", {})
        self.external_refs = self.data.get("external_refs", [])

    # ========================================================================
    # By Entity Type
    # ========================================================================

    def list_entities(self, entity_type: str) -> list[dict]:
        """
        List all unique entities of a given type.

        Args:
            entity_type: One of: creature, item, spell, action, background

        Returns:
            List of {name, identifier, mention_count, locations}
        """
        # Filter refs by type
        refs = [r for r in self.external_refs if r[0] == entity_type]

        # Merge by identifier
        entities = defaultdict(lambda: {"name": None, "mention_count": 0, "locations": []})

        for ref in refs:
            _, identifier, source_id, raw, display_name = ref
            entities[identifier]["name"] = display_name
            entities[identifier]["mention_count"] += 1
            entities[identifier]["identifier"] = identifier
            entities[identifier]["locations"].append(source_id)

        # Convert to list and sort by mention count
        result = []
        for identifier, info in entities.items():
            info["locations"] = list(set(info["locations"]))  # Dedupe
            result.append(info)

        return sorted(result, key=lambda x: x["mention_count"], reverse=True)

    def show_entities(self, entity_type: str, limit: int = 20) -> None:
        """Print entities of a type."""
        entities = self.list_entities(entity_type)

        print(f"\n=== @{entity_type.upper()} LIST ({len(entities)} unique) ===\n")
        print(f"{'Rank':<6} {'Name':<30} {'Mentions':<10} {'Locations'}")
        print("-" * 80)

        for i, entity in enumerate(entities[:limit], 1):
            name = entity["name"]
            count = entity["mention_count"]
            locs = entity["locations"]
            loc_str = ", ".join(locs[:3])
            if len(locs) > 3:
                loc_str += f" ... ({len(locs)} total)"

            print(f"{i:<6} {name:<30} {count:<10} {loc_str}")

        if len(entities) > limit:
            print(f"\n... and {len(entities) - limit} more")

    # ========================================================================
    # By Section
    # ========================================================================

    def get_section(self, section_id: str) -> dict | None:
        """Get full details of a section by ID."""
        return self.internal_index.get(section_id)

    def show_section(self, section_id: str, show_text: bool = False) -> None:
        """Print section details."""
        section = self.get_section(section_id)
        if not section:
            print(f"Section {section_id} not found")
            return

        print(f"\n=== SECTION: {section['name']} (id: {section_id}) ===")
        print(f"Type: {section['type']}")
        print(f"Page: {section['page']}")
        print(f"Parent: {section['parent_id']}")
        print(f"Children: {section['children']}")
        print(f"\nLinks: {section['internal_links_count']} internal, "
              f"{section['external_links_count']} external, "
              f"{section['mechanic_links_count']} mechanic")

        # External entities in this section
        if section['external_links']:
            print(f"\nExternal entities ({len(section['external_links'])}):")
            for link in section['external_links']:
                print(f"  - @{link['tag']}: {link['text']}")

        # Text content
        if show_text and section['text_content']:
            print(f"\nText content ({len(section['text_content'])} paragraphs):")
            for i, text in enumerate(section['text_content'][:10], 1):
                print(f"\n  [{i}] {text[:200]}{'...' if len(text) > 200 else ''}")
            if len(section['text_content']) > 10:
                print(f"\n  ... ({len(section['text_content']) - 10} more paragraphs)")

    def list_sections_by_type(self, section_type: str) -> list[dict]:
        """List all sections of a given type."""
        return [
            s for s in self.internal_index.values()
            if s['type'] == section_type
        ]

    # ========================================================================
    # Entity Locations
    # ========================================================================

    def find_entity_locations(self, entity_name: str) -> list[dict]:
        """
        Find all sections that mention an entity.

        Returns:
            List of {section_id, section_name, context}
        """
        locations = []

        for section_id, section in self.internal_index.items():
            for link in section['external_links']:
                if link['text'].lower() == entity_name.lower():
                    locations.append({
                        "section_id": section_id,
                        "section_name": section['name'],
                        "section_type": section['type'],
                        "page": section['page'],
                        "context": link['raw']
                    })

        return locations

    def show_entity_locations(self, entity_name: str) -> None:
        """Print all locations mentioning an entity."""
        locations = self.find_entity_locations(entity_name)

        print(f"\n=== LOCATIONS MENTIONING \"{entity_name}\" ({len(locations)}) ===\n")

        for loc in locations:
            print(f"[{loc['section_id']}] {loc['section_name']} ({loc['section_type']}, page {loc['page']})")
            print(f"  Context: {loc['context']}")

    # ========================================================================
    # Hierarchy
    # ========================================================================

    def show_hierarchy(self, section_id: str, max_depth: int = 3, indent: int = 0) -> None:
        """Print section hierarchy."""
        section = self.get_section(section_id)
        if not section:
            return

        prefix = "  " * indent
        print(f"{prefix}[{section['id']}] {section['name']} ({section['type']})")

        if indent < max_depth and section['children']:
            for child_id in section['children']:
                self.show_hierarchy(child_id, max_depth, indent + 1)

    def show_roots(self) -> None:
        """Show all root sections (no parent)."""
        roots = self.data.get('stats', {}).get('root_nodes', [])
        print(f"\n=== ROOT SECTIONS ({len(roots)}) ===\n")
        for root_id in roots:
            root = self.get_section(root_id)
            if root:
                print(f"[{root_id}] {root['name']} ({root['type']}, page {root['page']})")

    # ========================================================================
    # Summary Stats
    # ========================================================================

    def show_summary(self) -> None:
        """Print overall summary."""
        stats = self.data.get('stats', {})

        print("\n" + "=" * 60)
        print("ADVENTURE PARSER - SUMMARY")
        print("=" * 60)

        print(f"\nTotal structural nodes: {stats.get('total_nodes', 0)}")
        print(f"Total external references: {stats.get('total_external_refs', 0)}")

        print(f"\nEntity breakdown:")
        for entity_type, count in stats.get('entity_breakdown', {}).items():
            print(f"  @{entity_type}: {count}")

        print(f"\nRoot sections: {stats.get('root_nodes', [])}")

        # Section type breakdown
        type_counts = defaultdict(int)
        for section in self.internal_index.values():
            type_counts[section['type']] += 1

        print(f"\nSection types:")
        for section_type, count in sorted(type_counts.items()):
            print(f"  {section_type}: {count}")


# ========================================================================
# Convenience functions
# ========================================================================

def explore(parsed_data: Any) -> AdventureExplorer:
    """Create an explorer from parsed data or file path."""
    return AdventureExplorer(parsed_data)
