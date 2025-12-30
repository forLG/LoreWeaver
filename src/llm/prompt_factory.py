import re


class PromptFactory:
    # 预编译正则，匹配 "A1: Name", "B2 Name", "Area C3" 等格式
    AREA_PATTERN = re.compile(r"^(?:Area\s+)?([A-Z][0-9]+)[:\s]\s*(.+)$", re.IGNORECASE)

    @staticmethod
    def create_spatial_summary_prompt(node: dict, child_summaries: list[str]) -> str:
        """
        阶段一：递归生成纯文本的空间关系总结
        """
        title = node.get("title", "Untitled")
        content = node.get("content", "")

        # 基础 Prompt 结构
        return f"""/no_think
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
        return f"""/no_think
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

Known Location IDs (comma-separated list):
{location_list}

Section Info:
{section_context}

CRITICAL RULES:
1. Be SELECTIVE - Only map to locations that are EXPLICITLY mentioned or clearly described in the section.
2. Maximum 5 locations per section - prioritize the most relevant/primary locations.
3. If a location name appears in the text, map to that specific location ID.
4. If the section describes movement between locations, include all visited locations.
5. If the section is generic or doesn't clearly match any specific location, return an empty list.
6. DO NOT guess or infer locations not mentioned in the text.

Output JSON format: {{ "location_ids": ["location_id_1", "location_id_2"] }}
"""

    # Multi-Pass Extraction Prompts (for smaller models like qwen3-8b)
    @staticmethod
    def create_top_level_extraction_prompt(spatial_summary_text: str) -> str:
        """
        Multi-Pass Pass 1: Extract only top-level hierarchy (World, Region, Island, City level)
        Focus on establishing the foundational spatial structure without sub-locations.
        """
        return f"""
You are a D&D Knowledge Graph Architect. This is PASS 1 of a multi-pass extraction.

Input Text:
{spatial_summary_text}

TASK: Extract ONLY TOP-LEVEL locations (highest hierarchy levels).
- Level 1: World (e.g., "The Forgotten Realms")
- Level 2: Region/Coast (e.g., "The Sword Coast")
- Level 3: Major Locations (e.g., "Stormwreck Isle", "Neverwinter", "Mount Hotenow")

WHAT TO EXCLUDE:
- Do NOT extract sub-locations like rooms, caves, decks, or specific areas
- Do NOT extract features like statues, individual cells, or minor details
- Do NOT extract any location that is "part of" another location you're extracting

OUTPUT REQUIREMENTS:
- Use snake_case IDs (e.g., "the_forgotten_realms", "stormwreck_isle")
- Include type field (World, Region, Island, City, Mountain)
- Create "part_of" edges showing the hierarchy

Output Format (JSON ONLY):
{{
    "nodes": [
        {{ "id": "the_forgotten_realms", "label": "The Forgotten Realms", "type": "World" }},
        {{ "id": "the_sword_coast", "label": "The Sword Coast", "type": "Region" }},
        {{ "id": "stormwreck_isle", "label": "Stormwreck Isle", "type": "Island" }}
    ],
    "edges": [
        {{ "source": "the_sword_coast", "target": "the_forgotten_realms", "relation": "part_of" }},
        {{ "source": "stormwreck_isle", "target": "the_sword_coast", "relation": "part_of" }}
    ]
}}
"""

    @staticmethod
    def create_sub_location_extraction_prompt(parent_location: str, parent_label: str, spatial_summary_text: str, existing_nodes: str) -> str:
        """
        Multi-Pass Pass 2: Extract sub-locations for a specific parent location.
        The existing_nodes helps maintain ID consistency and avoid duplication.
        """
        return f"""
You are a D&D Knowledge Graph Architect. This is PASS 2 of a multi-pass extraction.

PARENT LOCATION: {parent_label} (id: {parent_location})

ALREADY EXTRACTED NODES (use these IDs for connections):
{existing_nodes}

Input Text:
{spatial_summary_text}

TASK: Extract ALL sub-locations within "{parent_label}".
This includes: buildings, rooms, caves, decks, areas, features - ANY location that is "part_of" {parent_location}.

MINIMUM DETAIL REQUIREMENTS:
- Every ship deck should be a separate node (C1, C2, C3, etc.)
- Every named room should be a separate node
- Every cave chamber should be a separate node (B1, B2, B3, etc.)
- Every NPC's personal space (cell, quarters) should be a separate node

RELATIONSHIP TYPES:
- "part_of": This location is inside the parent location
- "connected_to": Direct access between two locations
- "leads_to": Passage or path from one location to another

ID STANDARDIZATION:
- Use snake_case
- Use existing IDs from the list above when connecting to known locations
- Create new IDs for new locations following the same pattern

Output Format (JSON ONLY):
{{
    "nodes": [
        {{ "id": "seagrow_caves", "label": "Seagrow Caves Entrance", "type": "Caves" }},
        {{ "id": "main_deck", "label": "Main Deck (C1)", "type": "Deck" }}
    ],
    "edges": [
        {{ "source": "seagrow_caves", "target": "{parent_location}", "relation": "part_of", "desc": "Located on Stormwreck Isle" }},
        {{ "source": "glowing_fungus_tunnel", "target": "seagrow_caves", "relation": "part_of", "desc": "Within the cave system" }}
    ]
}}
"""

    @staticmethod
    def create_relationship_extraction_prompt(nodes_text: str, spatial_summary_text: str) -> str:
        """
        Multi-Pass Pass 3: Extract relationships between already-identified locations.
        Focus on connections that may have been missed in earlier passes.
        """
        return f"""
You are a D&D Knowledge Graph Architect. This is PASS 3 of a multi-pass extraction.

KNOWN LOCATIONS:
{nodes_text}

Input Text:
{spatial_summary_text}

TASK: Extract RELATIONSHIPS (edges) between the known locations above.
Focus on connections that show how locations relate to each other.

RELATIONSHIP TYPES:
- "connected_to": Direct access between locations (doors, passages, teleportation)
- "leads_to": Directional movement (paths, stairs, one-way passages)

DO NOT CREATE:
- New nodes (only use the known locations listed above)
- "part_of" edges (those were handled in earlier passes)

Output Format (JSON ONLY):
{{
    "edges": [
        {{ "source": "the_wreck_of_the_compass_rose", "target": "dragon_s_rest", "relation": "connected_to", "desc": "2.5 miles north, accessible by rowboat" }},
        {{ "source": "the_winch_house", "target": "the_shoreline_bluffs", "relation": "connected_to", "desc": "Connected by a path" }}
    ]
}}
"""

    @staticmethod
    def create_multi_pass_verification_prompt(full_graph_text: str, spatial_summary_text: str) -> str:
        """
        Multi-Pass Pass 4: Self-verification to check completeness.
        """
        return f"""
You are a Quality Assurance Expert for D&D Knowledge Graphs.

CURRENT GRAPH:
{full_graph_text}

SOURCE TEXT:
{spatial_summary_text}

TASK: Review the graph and answer these questions:
1. Are all major locations from the source text included?
2. Does every sub-location have a "part_of" relationship to its parent?
3. Are all IDs descriptive (not generic like "loc_1", "area_2")?
4. Are all ship deck sections (C1-C9) included as separate nodes?
5. Are all cave areas (B1-B6) included as separate nodes?

MISSING ITEMS CHECKLIST:
- Check for: decks, caves, rooms, cells, quarters, halls, passages
- Check for: named features like statues, altars, thrones
- Check for: outdoor areas like gardens, courtyards, cliffs

Output Format (JSON ONLY):
{{
    "is_complete": true/false,
    "missing_locations": ["list", "of", "missing", "locations"],
    "issues": ["description", "of", "issues", "found"],
    "revised_edges": [
        {{ "source": "id1", "target": "id2", "relation": "connected_to", "desc": "add missing edges here" }}
    ]
}}
"""

    @staticmethod
    def create_entity_enrichment_prompt(section_content: str, candidate_list: list, location_list: list) -> str:
        """
        阶段五：实体实例化与关系挖掘
        """
        return f"""/no_think
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
   - **Exception**: Always map "you", "characters", "party" to the ID "**characters**".
   - ID Format: Use the 'suggested_id' from candidates if available (snake_case, no prefix).

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
        {{ "id": "goblin_boss", "label": "Goblin Boss", "type": "Creature" }},
        {{ "id": "goblin_minion", "label": "Goblin Minion", "type": "Creature" }},
        {{ "id": "rusty_key", "label": "Rusty Key", "type": "Item" }},
        {{ "id": "iron_chest", "label": "Iron Chest", "type": "Location" }}
    ],
    "edges": [
        {{ "source": "goblin_boss", "target": "goblin_minion", "relation": "commands", "desc": "shouts orders to the minions" }},
        {{ "source": "rusty_key", "target": "iron_chest", "relation": "unlocks", "desc": "opens the locked chest in the corner" }},
        {{ "source": "goblin_boss", "target": "throne_room", "relation": "inhabits", "desc": "sits lazily on the throne" }}
    ]
}}
"""

    # ========================================================================
    # Entity-First Pipeline Prompts (for small models like qwen3-8b)
    # ========================================================================

    @staticmethod
    def create_ner_prompt(
        title: str,
        content: str,
        parent_context: str,  # noqa: ARG004 - passed for interface consistency
        known_entities: str
    ) -> str:
        """
        Entity-First Phase 1: Named Entity Recognition.

        Extract all named entities from text, using parent context for reference resolution.
        """
        return f"""/no_think
You are a D&D Entity Extractor. Extract named entities from the following text.

CURRENT SECTION: {title}

{known_entities}

TEXT TO PROCESS:
{content}

TASK: Extract ALL named entities mentioned in this text.

Entity Types to Extract:
- Locations: places, buildings, rooms, geographic features (e.g., "Dragon's Rest", "The Cave", "Cliffside Path")
- Creatures: monsters, NPCs, animals (e.g., "Runara", "Goblin Boss", "a zombie")
- Items: objects, equipment, treasures (e.g., "Rusty Key", "Magic Sword")
- Groups: organizations, parties, factions (e.g., "The Party", "Kobolds")

CRITICAL RULES:
1. **ID Format**: Use snake_case IDs (e.g., "dragon_s_rest", "runara", "rusty_key")

2. **Extract Both Named and Unnamed Entities**:
   - NAMED: "Dragon's Rest", "Runara" → extract with exact name
   - UNNAMED but RELEVANT: "a zombie", "two sailors", "the harbor", "a statue" → extract with descriptive ID
     - Use descriptive IDs: "zombie", "sailors", "harbor", "statue"
     - If multiple unnamed entities of same type exist, append numbers: "zombie_1", "zombie_2"
   - SKIP: Pure background/scenery with no relevance: "sunlight", "grass", "overcast sky"

3. **Reference Parent Entities**: If an entity mentioned here was already seen in parent sections, USE THE SAME ID from the known entities list above

4. **Add Aliases, Don't Duplicate**: If a generic term refers to a known entity, add it as an ALIAS instead of creating a new entity
   - Example: If parent has "cloister" and text mentions "the temple", add "temple" as an alias to cloister, don't create new "temple" entity

5. **Be Decisive**: Choose the best match and move on. Do not repeatedly reconsider your choices.

Output Format (JSON ONLY):
{{
    "entities": [
        {{"id": "dragon_s_rest", "label": "Dragon's Rest", "type": "Location", "aliases": ["temple", "monastery", "cloister"]}},
        {{"id": "runara", "label": "Runara", "type": "Creature", "aliases": ["bronze dragon", "elder"]}},
        {{"id": "zombie", "label": "Zombie", "type": "Creature", "aliases": []}},
        {{"id": "sailors", "label": "Sailors", "type": "Creature", "aliases": ["two sailors"]}},
        {{"id": "harbor", "label": "Harbor", "type": "Location", "aliases": ["calm harbor"]}},
        {{"id": "statue", "label": "Statue", "type": "Item", "aliases": ["towering statue"]}}
    ]
}}
"""

    @staticmethod
    def create_relation_extraction_prompt(
        title: str,
        content: str,
        entities_text: str
    ) -> str:
        """
        Entity-First Phase 3: Extract relations between known entities.

        Given a list of known entities, find relationships between them in the text.
        """
        return f"""/no_think
You are a D&D Relation Extractor. Extract relationships between known entities.

CURRENT SECTION: {title}

KNOWN ENTITIES:
{entities_text}

TEXT TO PROCESS:
{content}

TASK: Find relationships between the known entities mentioned in this text.

RELATIONSHIP TYPES:
- Spatial: "part_of" (inside), "connected_to" (adjacent), "leads_to" (path)
- Social: "commands" (authority), "serves" (loyalty), "allied_with" (friendship)
- State: "inhabits" (lives in), "guards" (protects), "stored_in" (contained)
- Action: "attacks" (hostile), "gives_quest_to" (interaction)

CRITICAL RULES:
1. **Only use entities from the KNOWN ENTITIES list above**
2. **Create edges only for relationships EXPLICITLY stated in the text**
3. **Include descriptions**: Add a "desc" field with context for each relation
4. **Be conservative**: Don't invent relationships that aren't clearly stated

Output Format (JSON ONLY):
{{
    "relations": [
        {{"source": "dragon_s_rest", "target": "stormwreck_isle", "relation": "part_of", "desc": "Located on the island"}},
        {{"source": "runara", "target": "dragon_s_rest", "relation": "inhabits", "desc": "Lives in the temple"}},
        {{"source": "goblin_boss", "target": "goblin_minion", "relation": "commands", "desc": "Leads the goblins"}}
    ]
}}
"""

    # ========================================================================
    # Natural Language Output Prompts (for small models like qwen3-8b)
    # More robust to truncation than JSON
    # ========================================================================

    @staticmethod
    def create_ner_prompt_natural(
        title: str,
        content: str,
        known_entities: str
    ) -> str:
        """
        Natural language NER prompt (no JSON required).

        Output format:
            Entity: Name
            Type: Location/Creature/Item
            ID: snake_case_id
            Aliases: alias1, alias2
        """
        return f"""
You are a D&D Entity Extractor. Extract named entities from the following text.

CURRENT SECTION: {title}

{known_entities}

TEXT TO PROCESS:
{content}

TASK: Extract ALL named entities mentioned in this text.

Entity Types to Extract:
- Locations: places, buildings, rooms, geographic features
- Creatures: monsters, NPCs, animals
- Items: objects, equipment, treasures
- Groups: organizations, parties, factions

CRITICAL RULES:
1. **ID Format**: Use snake_case IDs (e.g., dragon_s_rest, runara, rusty_key)

2. **Extract Both Named and Unnamed Entities**:
   - NAMED: "Dragon's Rest", "Runara" → extract with exact name
   - UNNAMED but RELEVANT: "a zombie", "two sailors", "the harbor", "a statue" → extract with descriptive ID
     - Use descriptive IDs: "zombie", "sailors", "harbor", "statue"
     - If multiple unnamed entities of same type exist, append numbers: "zombie_1", "zombie_2"
   - SKIP: Pure background/scenery with no relevance: "sunlight", "grass", "overcast sky"

3. **Reference Parent Entities**: If an entity was already seen in parent sections, USE THE SAME ID

4. **Add Aliases, Don't Duplicate**: If a generic term refers to a known entity, add it as an ALIAS instead of creating a new entity

5. **Be Decisive**: Choose the best match and move on. Do not repeatedly reconsider your choices.

OUTPUT FORMAT (one entity per block, separate blocks with blank line):
Entity: [name]
Type: [Location/Creature/Item/Group]
ID: [snake_case_id]
Aliases: [alias1, alias2, alias3]

[Repeat for each entity]
"""

    @staticmethod
    def create_relation_extraction_prompt_natural(
        title: str,
        content: str,
        entities_text: str
    ) -> str:
        """
        Natural language relation extraction prompt (no JSON required).
        """
        return f"""
You are a D&D Relation Extractor. Extract relationships between known entities.

CURRENT SECTION: {title}

KNOWN ENTITIES:
{entities_text}

TEXT TO PROCESS:
{content}

TASK: Find relationships between the known entities mentioned in this text.

RELATIONSHIP TYPES:
- Spatial: part_of (inside), connected_to (adjacent), leads_to (path)
- Social: commands (authority), serves (loyalty), allied_with (friendship)
- State: inhabits (lives in), guards (protects), stored_in (contained)
- Action: attacks (hostile), gives_quest_to (interaction)

CRITICAL RULES:
1. **Only use entities from the KNOWN ENTITIES list above**
2. **Create edges only for relationships EXPLICITLY stated in the text**
3. **Include descriptions** with context for each relation
4. **Be conservative**: Don't invent relationships

OUTPUT FORMAT (one relation per block, separate blocks with blank line):
Relation: [relation_type]
Source: [entity_id]
Target: [entity_id]
Description: [context description]

[Repeat for each relationship]
"""

    @staticmethod
    def create_entity_resolution_prompt_natural(node_list_text: str) -> str:
        """
        Natural language entity resolution prompt (no JSON required).
        """
        return f"""
You are a Data Deduplication Expert for a Knowledge Graph.

Below is a list of raw nodes extracted from different chapters of a D&D book.
Many nodes refer to the SAME location but have different IDs or slightly different names.

RAW NODE LIST:
{node_list_text}

TASK: Identify duplicates and map them to a single CANONICAL ID.

RULES:
- If "dragon_s_rest" and "dragons_rest_ch1" are the same place, map: dragans_rest_ch1 -> dragon_s_rest
- If "A1" and "area_a1" are the same, map: area_a1 -> A1
- Prefer shorter, cleaner IDs (e.g., "A1" over "area_a1_cliff")
- Ignore distinct locations (do not merge "A1" and "A2")

OUTPUT FORMAT:
[duplicate_id] -> [canonical_id]

EXAMPLE:
dragon_rest -> dragon_s_rest
area_a1 -> A1
the_beach -> rocky_shore

[Output mappings below, one per line]
"""
