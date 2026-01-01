"""
Adventure Parser - 2-Stage Pipeline for Knowledge Graph Prep

Stage 1: Build Internal Index
    - Extract all structural nodes (section, entries, inset, insetReadaloud)
    - Capture hierarchy (parent-child relationships)
    - Extract and clean text content from each node

Stage 2: Extract External References
    - Find all {@creature}, {@item}, {@spell} references
    - Link them to the section IDs where they appear

Output:
    - internal_index: Dict[ID] -> full node info with text_content
    - external_refs: List of (entity_type, identifier, source_id)
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InternalNode:
    """A structural node from the adventure (Stage 1 output)."""
    id: str
    type: str              # section, entries, inset, insetReadaloud
    name: str
    page: int | None
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    text_content: list[str] = field(default_factory=list)

    # Links found in this node (categorized)
    internal_links: list[dict] = field(default_factory=list)  # {@area, @adventure, @book}
    external_links: list[dict] = field(default_factory=list)  # {@creature, @item, @spell}
    mechanic_links: list[dict] = field(default_factory=list)  # {@dc, @dice, @skill}


# Tag categorization
ENTITY_TAGS = {'creature', 'item', 'spell', 'action', 'background'}
NAVIGATION_TAGS = {'area', 'adventure', 'book', 'quickref', '5etoolsImg'}
MECHANIC_TAGS = {'dc', 'dice', 'skill', 'condition', 'sense'}
FORMAT_TAGS = {'b', 'i', 'note', 'bold', 'italic'}


class AdventureParser:
    """
    2-stage parser for adventure JSON.

    Stage 1: Build internal index with text content
    Stage 2: Extract external entity references
    """

    def __init__(self):
        self.internal_index: dict[str, InternalNode] = {}
        self.external_refs: list[tuple] = []  # (entity_type, identifier, source_id)

    def parse(self, adventure_data: dict) -> dict:
        """
        Main entry point. Parse adventure JSON with 2-stage pipeline.

        Args:
            adventure_data: The parsed adventure-dosi.json content

        Returns:
            {
                "internal_index": {id: {type, name, page, parent, children, text_content, links}},
                "external_refs": [(entity_type, identifier, source_id, raw_text, display_name)],
                "stats": {...}
            }
        """
        # Reset state
        self.internal_index = {}
        self.external_refs = []

        data_entries = adventure_data.get("data", [])

        # ============================================================
        # STAGE 1: Build Internal Index (structure + text content)
        # ============================================================
        self._build_internal_index(data_entries)

        # ============================================================
        # STAGE 2: Extract External References
        # ============================================================
        self._extract_external_references()

        # Convert to dict for JSON output
        return self._to_dict()

    # ========================================================================
    # STAGE 1: Internal Index
    # ========================================================================

    def _build_internal_index(self, entries: list, parent_id: str = None, depth: int = 0) -> None:
        """
        Stage 1: Recursively build index of all structural nodes.

        For each node, extract:
        - Metadata (id, type, name, page, parent)
        - Hierarchy (children)
        - Text content (cleaned of tags)
        - Links (categorized)
        """
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type")
            entry_id = entry.get("id")

            # Check if this is a structural node
            is_structural = entry_id and entry_type in {
                "section", "entries", "inset", "insetReadaloud"
            }

            if is_structural:
                # Create node
                node = InternalNode(
                    id=entry_id,
                    type=entry_type,
                    name=entry.get("name", ""),
                    page=entry.get("page"),
                    parent_id=parent_id
                )

                # Link to parent
                if parent_id and parent_id in self.internal_index:
                    self.internal_index[parent_id].children.append(entry_id)

                # Process entries array
                if "entries" in entry:
                    self._process_entry_content(entry["entries"], node)

                self.internal_index[entry_id] = node

                # Recurse into children
                self._build_internal_index(entry["entries"], entry_id, depth + 1)

            else:
                # Non-structural node, continue traversing
                if isinstance(entry, dict) and "entries" in entry:
                    self._build_internal_index(entry["entries"], parent_id, depth)

    def _process_entry_content(self, entries: list, node: InternalNode) -> None:
        """
        Process the entries array of a node, extracting text and links.

        Handles:
        - Plain text strings
        - Nested structural nodes (recursion)
        - Lists, tables, images, etc.
        """
        for entry in entries:
            if isinstance(entry, str):
                # Plain text - extract and clean
                clean_text, links = self._parse_text_and_links(entry)
                if clean_text:
                    node.text_content.append(clean_text)
                # Categorize links
                for link in links:
                    self._categorize_link(link, node)

            elif isinstance(entry, dict):
                entry_type = entry.get("type")

                # Nested structural node - will be processed separately in recursion
                if entry.get("id") and entry_type in {"section", "entries", "inset", "insetReadaloud"}:
                    continue

                # Handle different content types
                if entry_type == "list":
                    self._process_list(entry, node)
                elif entry_type == "table":
                    self._process_table(entry, node)
                elif entry_type == "image":
                    self._process_image(entry, node)
                elif entry_type == "gallery":
                    self._process_gallery(entry, node)
                elif entry_type == "quote":
                    self._process_quote(entry, node)
                elif "entries" in entry:
                    # Nested entries (not structural)
                    self._process_entry_content(entry["entries"], node)

    def _process_list(self, entry: dict, node: InternalNode) -> None:
        """Process list entries."""
        items = entry.get("items", [])
        for item in items:
            if isinstance(item, str):
                clean_text, links = self._parse_text_and_links(item)
                node.text_content.append(f"- {clean_text}")
                for link in links:
                    self._categorize_link(link, node)
            elif isinstance(item, dict):
                name = item.get("name", "")
                item_entry = item.get("entry", "")
                if name and item_entry:
                    clean_entry, links = self._parse_text_and_links(item_entry)
                    node.text_content.append(f"- **{name}**: {clean_entry}")
                    for link in links:
                        self._categorize_link(link, node)
                elif item_entry:
                    clean_entry, links = self._parse_text_and_links(item_entry)
                    node.text_content.append(f"- {clean_entry}")
                    for link in links:
                        self._categorize_link(link, node)

    def _process_table(self, entry: dict, node: InternalNode) -> None:
        """Process table entries."""
        if "caption" in entry:
            node.text_content.append(f"**Table: {entry['caption']}**")
        for row in entry.get("rows", []):
            if isinstance(row, list):
                row_text = " | ".join(str(cell) for cell in row)
                node.text_content.append(row_text)

    def _process_image(self, entry: dict, node: InternalNode) -> None:
        """Process image entry."""
        title = entry.get("title", "")
        image_type = entry.get("imageType", "")
        if title:
            if image_type:
                node.text_content.append(f"[Image ({image_type}): {title}]")
            else:
                node.text_content.append(f"[Image: {title}]")

    def _process_gallery(self, entry: dict, node: InternalNode) -> None:
        """Process gallery of images."""
        node.text_content.append("**Gallery:**")
        for img in entry.get("images", []):
            title = img.get("title", "")
            image_type = img.get("imageType", "")
            if title:
                if image_type:
                    node.text_content.append(f"[Image ({image_type}): {title}]")
                else:
                    node.text_content.append(f"[Image: {title}]")

    def _process_quote(self, entry: dict, node: InternalNode) -> None:
        """Process quote/blockquote entries."""
        for line in entry.get("entries", []):
            if isinstance(line, str):
                clean_text, links = self._parse_text_and_links(line)
                node.text_content.append(f"> {clean_text}")
                for link in links:
                    self._categorize_link(link, node)

    def _parse_text_and_links(self, text: str) -> tuple[str, list[dict]]:
        """
        Parse text, extracting links and cleaning markup.

        Returns:
            (cleaned_text, extracted_links)
        """
        if not isinstance(text, str):
            return "", []

        links = []
        pattern = re.compile(r'\{@(\w+)\s+([^{}]+)\}')

        def replace_tag(match):
            raw = match.group(0)
            tag_type = match.group(1)
            content = match.group(2)

            parts = content.split('|')
            display_text = parts[0]
            attributes = parts[1:]

            # Extract link info
            link_info = {
                "tag": tag_type,
                "text": display_text,
                "attrs": attributes,
                "raw": raw
            }
            links.append(link_info)

            return display_text

        # Apply replacements
        cleaned = pattern.sub(replace_tag, text)
        return cleaned, links

    def _categorize_link(self, link: dict, node: InternalNode) -> None:
        """Categorize a link and add to appropriate list."""
        tag_type = link["tag"]

        if tag_type in ENTITY_TAGS:
            node.external_links.append(link)
        elif tag_type in NAVIGATION_TAGS:
            node.internal_links.append(link)
        elif tag_type in MECHANIC_TAGS:
            node.mechanic_links.append(link)
        # FORMAT_TAGS are ignored (text styling only)

    # ========================================================================
    # STAGE 2: External References
    # ========================================================================

    def _extract_external_references(self) -> None:
        """
        Stage 2: Extract all external entity references.

        Output: self.external_refs as list of tuples:
            (entity_type, identifier, source_id, raw_text, display_name)
        """
        for node_id, node in self.internal_index.items():
            for link in node.external_links:
                entity_type = link["tag"]
                display_name = link["text"]
                attrs = link["attrs"]

                # Build identifier
                if entity_type == "creature" and attrs:
                    source = attrs[0]
                    identifier = f"{display_name}|{source}"
                elif entity_type in {"item", "spell"} and attrs:
                    source = attrs[0]
                    identifier = f"{display_name}|{source}"
                else:
                    identifier = display_name

                self.external_refs.append((
                    entity_type,
                    identifier,
                    node_id,
                    link["raw"],
                    display_name
                ))

    # ========================================================================
    # Output
    # ========================================================================

    def _to_dict(self) -> dict:
        """Convert results to dict for JSON serialization."""
        # Convert internal nodes to dicts
        internal_dict = {}
        for node_id, node in self.internal_index.items():
            internal_dict[node_id] = {
                "id": node.id,
                "type": node.type,
                "name": node.name,
                "page": node.page,
                "parent_id": node.parent_id,
                "children": node.children,
                "text_content": node.text_content,
                "internal_links_count": len(node.internal_links),
                "external_links_count": len(node.external_links),
                "mechanic_links_count": len(node.mechanic_links),
                "external_links": node.external_links,  # Include for reference
            }

        # Stats
        stats = {
            "total_nodes": len(self.internal_index),
            "total_external_refs": len(self.external_refs),
            "entity_breakdown": self._get_entity_breakdown(),
            "root_nodes": [nid for nid, n in self.internal_index.items() if n.parent_id is None]
        }

        return {
            "internal_index": internal_dict,
            "external_refs": self.external_refs,
            "stats": stats
        }

    def _get_entity_breakdown(self) -> dict[str, int]:
        """Count external refs by type."""
        breakdown = defaultdict(int)
        for ref in self.external_refs:
            breakdown[ref[0]] += 1
        return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))


def parse_adventure(file_path: str, output_path: str = None) -> dict:
    """
    Convenience function: Parse adventure JSON file.

    Args:
        file_path: Path to adventure-dosi.json
        output_path: Optional path to save results

    Returns:
        Same as AdventureParser.parse()
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parser = AdventureParser()
    result = parser.parse(data)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    return result
