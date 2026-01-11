import re
from typing import Dict, List

class PromptFactory:
    """Factory for creating prompts used in various stages of the knowledge graph construction."""

    # Compiled regex pattern matching area formats like "A1: Name", "B2 Name", "Area C3".
    AREA_PATTERN = re.compile(r"^(?:Area\s+)?([A-Z][0-9]+)[:\s]\s*(.+)$", re.IGNORECASE)

    @staticmethod
    def create_spatial_summary_prompt(node: Dict, child_summaries: List[str]) -> str:
        """
        Stage 1: Recursively generate a plain text summary of spatial relationships.
        
        Args:
            node: The dictionary representing the current node (e.g., a section).
            child_summaries: A list of spatial summaries from child nodes.

        Returns:
            A string containing the prompt for generating the spatial report.
        """
        title = node.get("title", "Untitled")
        content = node.get("content", "")

        return f"""
You are a D&D Cartographer. Generate a SPATIAL REPORT for the node: "{title}".

Current Text:
{content}

Sub-area Reports (from children nodes):
{child_summaries}

Task:
1. **STRICTLY Filter Non-Locations**: 
   - **ONLY** physical locations (Rooms, Buildings, Islands, Caves, Landmarks, etc) can be nodes in the hierarchy.
   - **EXCLUDE** abstract concepts, narratives, or containers such as: "Quests", "Events", "Encounters", "Chapters", "Introductions", "Welcome to...".
   - If the current node "{title}" is NOT a physical location (e.g., "Cloister Quests"), **DO NOT** create a hierarchy node for it. Instead, **PROMOTE** (hoist) its valid spatial children to the top level of the list.
   - **Synthesize Names**: If the text describes a valid physical location or spatial relationship but lacks a specific proper name (or the title is abstract), **CREATE** a descriptive name yourself based on the content

2. **Refine & Flatten Hierarchy**:
   - **Correction**: If a child report lists a major location under a non-spatial parent, move it to the top level.
   - **Siblings vs Children**: Major adventure sites are usually **siblings**, not inside each other. Ensure they are listed as separate top-level items unless the text explicitly says one is inside the other.
   - **Deduplicate**: If the same location appears multiple times with slightly different names, **MERGE** them into a single node using the most specific or canonical name.

3.  **Preserve IDs**: Keep all Area IDs (e.g., A1, B2) visible.

Output Requirements:
- If no spatial info exists anywhere (and no valid children), output: "NO_SPATIAL_INFO"
- Otherwise, use this format:
  **[Overview]**: <Brief description of recent spatial hierarchy.>
  **[Direct Connections]**: <Specific exits mentioned in THIS node's text>
  **[Spatial Hierarchy]**:
    - Valid Location A: <Connection info>
      - Sub-Location A1: <Connection info>
    - Valid Location B (Promoted/Hoisted if necessary): <Connection info>
"""

    @staticmethod
    def create_graph_extraction_prompt(spatial_summary_text: str) -> str:
        """
        Stage 2: Extract a JSON Knowledge Graph from the single-chapter spatial summary text.
        
        Args:
            spatial_summary_text: The spatial summary text generated in Stage 1.

        Returns:
            A string containing the prompt for extracting the knowledge graph.
        """
        return f"""
You are a D&D Knowledge Graph Architect.
Extract a Location Knowledge Graph (JSON) from the provided text.

Input Text:
{spatial_summary_text}

Rules for Extraction:
1. **Nodes (Locations)**:
   - Extract all physical locations.
   - **CRITICAL - ID Standardization**: 
     - You MUST use **snake_case** for IDs to ensure consistency.
     - Example: "Dragon's Rest" -> "dragons_rest", "Area A1" -> "a1", "The Beach" -> "the_beach".
     - Do not use specific code (A1, B2) as ID or label.

2. **Edges (Relationships)**:
   - **"part_of"**: Hierarchical containment.
   - **"connected_to"**: General connection.
   - **"leads_to"**: Directional movement (paths, stairs).

Output Format (JSON ONLY):
{{
    "nodes": [
        {{ "id": "dragons_rest", "label": "Dragon's Rest", "type": "location" }},
        {{ "id": "a1", "label": "Cliffside Path (A1)", "type": "location" }}
    ],
    "edges": [
        {{ "source": "dragons_rest", "target": "stormreck_isle",  "relationship": "part_of", "desc":  Dragon's rest is part of the stormreck islend." }},
        {{ "source": "doorway", "target": "cells", "relationship": "leads_to", "desc": "The doorway leads to the cells " }}
    ]
}}
"""

    @staticmethod
    def create_entity_resolution_prompt(node_list_text: str) -> str:
        """
        Stage 3: Prompt for entity deduplication and alignment.

        Args:
            node_list_text: A raw list of nodes extracted from different chapters.
        
        Returns:
             A string containing the prompt for entity resolution.
        """
        return f"""
You are a Data Deduplication Expert for a Knowledge Graph.
Below is a list of raw nodes extracted from different chapters of a D&D book.
Many nodes refer to the SAME location but have different IDs or slightly different names.

Raw Node List:
{node_list_text}

Task:
Identify duplicates and map them to a single CANONICAL ID.
- Use more typical, cleaner ID, like mapping "dragons_rest_chositer" -> "dragons_rest".
- Ignore distinct locations (do not merge "A1" and "A2").

Output Format (JSON ONLY):
Return a dictionary where KEY is the duplicate ID and VALUE is the canonical ID.
Only include IDs that need to change.
{{
    "dragons_rest_ch1": "dragons_rest",
    "the_beach": "rocky_shore"
}}
"""

    @staticmethod
    def create_section_mapping_prompt(section_context: str, location_list: str) -> str:
        """
        Stage 4: Map the content of a section to specific Location IDs.

        Args:
            section_context: The text content of the section.
            location_list: The list of known Location IDs.

        Returns:
            A string containing the prompt for section mapping.
        """
        return f"""
You are a D&D Knowledge Graph Assistant.
I have a section from a book and a list of known Location IDs extracted from the same book.
Task: Map the Section to the most likely Location ID(s) where the events in that section take place.

Known Location IDs:
{location_list}

Section Info:
{section_context}

Rules:
1. If the section clearly describes events inside one or more of the Known Locations, map it.
2. If the section is generic or doesn't match any specific location, return an empty list.
3. Output JSON format: {{ "location_ids": ["location_id_1", "location_id_2"] }}
"""

    @staticmethod
    def create_entity_enrichment_prompt(section_content: str, candidate_list: list, location_list: list) -> str:
        """
        Stage 5: Entity instantiation and relationship extraction.

        Args:
            section_content: The text content of the section.
            candidate_list: A list of candidate entities (metdata) that might appear.
            location_list: The list of locations relevant to this context.

        Returns:
            A string containing the prompt for entity enrichment.
        """
        return f"""
You are a D&D Knowledge Graph Builder.
Context: The following text describes events occurring at these Location(s): [{location_list}].

Candidate Entities (from metadata):
{candidate_list}

Input Text:
{section_content}

Task:
1. **Instantiate Entities**: Identify which "Candidate Entities" are actually present or mentioned.
   - You must **ONLY** use Node IDs provided in the "Candidate Entities" list or the "Context Locations" list.
   - **DO NOT** invent new Node IDs.
   - If the text mentions an entity (e.g., "a mysterious guard") but it is NOT in the candidate list, **IGNORE IT**. Do not create a node for it.
   - **Exception**: Always map "you", "characters", "party" to the ID "**players**".
   - Decide the type of the node, choose between "location", "npc", "monster", "item" and "player".
   - ID Format: Use the 'suggested_id' from candidates if available.

2. **Extract Relations (Edges)**:
   - **DO NOT** extract relationships between two Location nodes (e.g., "connected_to", "part_of"). Spatial topology is already handled.
   - **CRITICAL PRIORITY**: Focus on the **Ecology and State of the World** first. 
     - How do NPCs relate to each other? (e.g., leader/minion, rivals)
     - Where are items physically located? (e.g., inside a chest, worn by a statue)
     - What are monsters doing in the location? (e.g., sleeping, guarding)
   - **Secondary Priority**: Player interactions. Only record SIGNIFICANT interactions (e.g., Boss fights, Quest giving), ignore trivial observations (e.g., "party sees wall").

   - **Recommended Verbs**:
     - **Social/Political**: `commands`, `serves`, `worships`, `allied_with`, `rival_of`.
     - **Spatial/State**: `inhabits`, `stored_in` (for items in containers), `hidden_at`, `locks`, `unlocks`.
     - **Action**: `guards`, `patrols`, `attacks`, `gives_quest_to`.

   - **Description (REQUIRED)**: Add a `desc` field to EVERY edge summarizing the context.

Output JSON Format:
{{
    "nodes": [
        {{ "id": "goblin_boss", "label": "Goblin Boss", "type": "monster" }},
        {{ "id": "merchant goblin", "label": "Goblin Minion", "type": "npc" }},
        {{ "id": "rusty_key", "label": "Rusty Key", "type": "item" }},
        {{ "id": "iron_chest", "label": "Iron Chest", "type": "location" }}
    ],
    "edges": [
        {{ "source": "goblin_boss", "target": "goblin_minion", "relationship": "commands", "desc": "shouts orders to the minions" }},
        {{ "source": "rusty_key", "target": "iron_chest", "relationship": "unlocks", "desc": "opens the locked chest in the corner" }},
        {{ "source": "goblin_boss", "target": "throne_room", "relationship": "inhabits", "desc": "sits lazily on the throne" }}
    ]
}}
"""