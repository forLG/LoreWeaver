import json
from typing import Dict, List, Any, Optional
from src.utils.link import LinkProcessor
from src.utils.logger import logger

class ShadowNode:
    """Represents a node in the shadow tree structure.

    Attributes:
        id: The unique identifier for the node.
        title: The title of the node.
        type: The type of the node (e.g., section, entry).
        text_content: List of text paragraphs accumulated for this node.
        links: List of links found in the accumulated text.
        children: List of child ShadowNodes.
    """

    def __init__(self, id: str, title: str, node_type: str):
        self.id = id
        self.title = title
        self.type = node_type
        self.text_content: List[str] = []
        self.links: List[Dict] = []
        self.children: List['ShadowNode'] = []

    def add_text(self, text: str):
        """Adds text to the node, parsing links and cleaning content.

        Args:
            text: The text string to add.
        """
        if not text:
            return
        clean_text, links = LinkProcessor.parse_and_clean(text)
        self.text_content.append(clean_text)
        self.links.extend(links)

    def to_dict(self):
        """Converts the node and its children to a dictionary format.

        Performs final deduplication of links before outputting. This is done
        here rather than in add_text to avoid performance overhead during accumulation.

        Returns:
            A dictionary representation of the ShadowNode.
        """
        unique_links = []
        seen = set()
        for link in self.links:
            link_signature = json.dumps(link, sort_keys=True)
            if link_signature not in seen:
                seen.add(link_signature)
                unique_links.append(link)

        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "content": "\n".join(self.text_content),
            "links": unique_links,
            "children": [child.to_dict() for child in self.children]
        }

class ShadowTreeBuilder:
    """Builder class for constructing a shadow tree from source data."""

    def build(self, data: List[Dict]) -> List[Dict]:
        """Builds the entire shadow tree from the provided data.

        Args:
            data: A list of data entries to process.

        Returns:
            A list of dictionary representations of the root nodes.
        """
        roots = []
        for entry in data:
            node = self._process_entry(entry)
            if node:
                roots.append(node.to_dict())
        return roots

    def _process_entry(self, entry: Any) -> Optional[ShadowNode]:
        """Recursive entry point for processing data entries.

        Creates a ShadowNode if the entry is a structural node (e.g., Section),
        or extracts content if it is a content block.

        Strategies:
        - Section/Inset/Entries types -> Create new node, recurse children.
        - String/List/Image types -> Content belongs to current context, extract content.

        Args:
            entry: The data entry to process.

        Returns:
            A ShadowNode if a new node is created, otherwise None.
        """
        
        # Create new ShadowNode for structural nodes
        # TODO: Support more structural types as needed
        if isinstance(entry, dict) and entry.get("type") in ["section", "entries", "inset", "insetReadaloud"]:
            node_id = entry.get("id", "")
            title = entry.get("name", "Untitled Section")
            node_type = entry.get("type")
            
            node = ShadowNode(node_id, title, node_type)
            
            # Process entries under this node
            if "entries" in entry:
                for child_entry in entry["entries"]:
                    # Recursively check child items
                    if self._is_structural_node(child_entry):
                        # If child is structural, add to children
                        child_node = self._process_entry(child_entry)
                        if child_node:
                            node.children.append(child_node)
                    else:
                        # If child is content, extract to current node
                        self._extract_content(child_entry, node)
            
            return node
            
        return None

    def _is_structural_node(self, entry: Any) -> bool:
        """Determines if an entry should be a standalone child node.

        Args:
            entry: The entry to check.

        Returns:
            True if structural, False otherwise.
        """
        if isinstance(entry, dict):
            # These types usually contain significant content and map to independent nodes
            return entry.get("type") in ["section", "entries", "inset", "insetReadaloud"]
        return False

    def _extract_content(self, entry: Any, current_node: ShadowNode):
        """Extracts content from non-structural nodes and merges it into the current node.

        Args:
            entry: The entry to extract content from.
            current_node: The node to add content to.
        """
        
        # Case 1: Pure string
        if isinstance(entry, str):
            current_node.add_text(entry)
            
        # Case 2: Dictionary types (List, Image, Table, Quote, etc.)
        elif isinstance(entry, dict):
            entry_type = entry.get("type")
            
            if entry_type == "list":
                # Process list items
                for item in entry.get("items", []):
                    if isinstance(item, str):
                        current_node.add_text(f"- {item}")
                    elif isinstance(item, dict) and "entry" in item:
                        # Process list items with titles (name: entry)
                        name = item.get("name", "")
                        text = item.get("entry", "")
                        full_text = f"- **{name}**: {text}" if name else f"- {text}"
                        current_node.add_text(full_text)
            
            elif entry_type == "table":
                # Simple table processing: extract caption and rows
                if "caption" in entry:
                    current_node.add_text(f"**Table: {entry['caption']}**")
                for row in entry.get("rows", []):
                    # Row might be a list of strings
                    row_text = " | ".join([str(cell) for cell in row])
                    current_node.add_text(row_text)

            elif entry_type == "image":
                # Only process images with titles (usually maps or important diagrams)
                title = entry.get("title", "")
                image_type = entry.get("imageType", "")
                if title:
                    if image_type:
                        current_node.add_text(f"[Image ({image_type}): {title}]")
                    else: 
                        current_node.add_text(f"[Image: {title}]")

            elif entry_type == "gallery":
                # Process gallery, only if valid images exist
                is_valid = False
                for img in entry.get("images", []):
                    title = img.get("title", "")
                    if title:
                        is_valid = True
                        break

                if is_valid:
                    current_node.add_text("**Gallery:**")
                    for img in entry.get("images", []):
                        title = img.get("title", "")
                        image_type = img.get("imageType", "")
                        if title:
                            if image_type:
                                current_node.add_text(f"[Image ({image_type}): {title}]")
                            else: 
                                current_node.add_text(f"[Image: {title}]")
            
            elif entry_type == "quote":
                # Blockquote
                for line in entry.get("entries", []):
                    if isinstance(line, str):
                        current_node.add_text(f"> {line}")

            else:
                # Unknown type, log warning
                logger.warning(f"Unknown entry type encountered: '{entry_type}' in node '{current_node.title}' (ID: {current_node.id})")