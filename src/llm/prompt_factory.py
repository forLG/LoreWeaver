import re
from typing import Dict, List

class PromptFactory:
    # 预编译正则，匹配 "A1: Name", "B2 Name", "Area C3" 等格式
    AREA_PATTERN = re.compile(r"^(?:Area\s+)?([A-Z][0-9]+)[:\s]\s*(.+)$", re.IGNORECASE)

    @staticmethod
    def create_spatial_summary_prompt(node: Dict, child_summaries: List[str]) -> str:
        """
        阶段一：递归生成纯文本的空间关系总结
        """
        title = node.get("title", "Untitled")
        content = node.get("content", "")
        
        # 基础 Prompt 结构
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
        阶段二：从单章总结文本中提取 JSON
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
     - If a specific code exists (A1, B2), use it as the primary ID (lowercase).

2. **Edges (Relationships)**:
   - **"part_of"**: Hierarchical containment.
   - **"connected_to"**: General connection.
   - **"leads_to"**: Directional movement (paths, stairs).

Output Format (JSON ONLY):
{{
    "nodes": [
        {{ "id": "dragons_rest", "label": "Dragon's Rest", "type": "Region" }},
        {{ "id": "a1", "label": "Cliffside Path (A1)", "type": "Path" }}
    ],
    "edges": [
        {{ "source": "a1", "target": "dragons_rest", "relation": "part_of" }},
        {{ "source": "a1", "target": "a2", "relation": "leads_to", "desc": "North path" }}
    ]
}}
"""
    
    @staticmethod
    def create_entity_resolution_prompt(node_list_text: str) -> str:
        """
        阶段三：实体对齐 Prompt
        """
        return f"""
You are a Data Deduplication Expert for a Knowledge Graph.
Below is a list of raw nodes extracted from different chapters of a D&D book.
Many nodes refer to the SAME location but have different IDs or slightly different names.

Raw Node List:
{node_list_text}

Task:
Identify duplicates and map them to a single CANONICAL ID.
- If "dragon_s_rest" and "dragons_rest_ch1" are the same place, map "dragons_rest_ch1" -> "dragon_s_rest".
- If "A1" and "area_a1" are the same, map "area_a1" -> "A1".
- Prefer shorter, cleaner IDs (e.g., "A1" over "area_a1_cliff").
- Ignore distinct locations (do not merge "A1" and "A2").

Output Format (JSON ONLY):
Return a dictionary where KEY is the duplicate ID and VALUE is the canonical ID.
Only include IDs that need to change.
{{
    "dragons_rest_ch1": "dragons_rest",
    "area_a1": "A1",
    "the_beach": "rocky_shore"
}}
"""

    @staticmethod
    def create_section_mapping_prompt(section_context: str, location_list: str) -> str:
        """
        阶段四：映射章节至地点 ID
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