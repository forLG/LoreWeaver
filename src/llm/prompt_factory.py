import re
from typing import Dict, List

class PromptFactory:
    # 预编译正则，匹配 "A1: Name", "B2 Name", "Area C3" 等格式
    AREA_PATTERN = re.compile(r"^(?:Area\s+)?([A-Z][0-9]+)[:\s]\s*(.+)$", re.IGNORECASE)

    @staticmethod
    def create_prompt(node: Dict, child_summaries: List[str] = None) -> str:
        node_type = node.get("type", "section")
        content = node.get("content", "")
        title = node.get("title", node.get("name", "Untitled"))
        links = [link['text'] for link in node.get("links", [])]
        
        # 基础上下文 (针对文本类节点)
        base_context = f"Title: {title}\nContent: {content}\nEntities: {', '.join(links)}"
        children_text = "\n".join(child_summaries) if child_summaries else "None"

        # 优先检查是否为区域节点
        area_match = PromptFactory.AREA_PATTERN.match(title)
        if area_match:
            area_id = area_match.group(1) # e.g., "A1"
            area_name = area_match.group(2) # e.g., "Path and Monastic Cells"
            return PromptFactory._prompt_for_area(base_context, children_text, area_id, area_name)

        # 根据类型分发
        # TODO: 支持更多类型
        if node_type == "section":
            return PromptFactory._prompt_for_section(base_context, children_text)
        elif node_type == "entries":
            return PromptFactory._prompt_for_entries(base_context, children_text)
        elif node_type in ["inset", "insetReadaloud"]:
            return PromptFactory._prompt_for_inset(base_context, node_type)
        
        # 下面三种形式的节点目前还不支持
        # 在 `src/builder/shadow_builder.py` 中，
        # 这三种节点中的文字描述会被提取出来并嵌入父节点的文字描述
        # TODO: 后续可能需要支持图片识别
        elif node_type == "image":
            return PromptFactory._prompt_for_image(node)
        elif node_type == "gallery":
            return PromptFactory._prompt_for_gallery(node)
        elif node_type == "list":
            return PromptFactory._prompt_for_list(base_context, node)
        else:
            return PromptFactory._prompt_default(base_context, children_text)
        
    @staticmethod
    def _prompt_for_area(context: str, children_text: str, area_id: str, area_name: str) -> str:
        return f"""
You are analyzing a specific MAP LOCATION (Area {area_id}: {area_name}) in a D&D adventure.
Your goal is to extract spatial data and gameplay elements.

Context:
{context}

Child Summaries (Sub-areas or details):
{children_text}

Task:
1. **Spatial Connections**: Explicitly list any exits or connections to other areas (e.g., "Stairs lead up to A2", "A door connects to C4"). If mentioned, note the direction (North, South, etc.).
2. **Environment**: Briefly describe the sensory atmosphere (lighting, smells, sounds) derived from the text or read-aloud sections.
3. **Key Features**: List major interactive objects (statues, traps, furniture) or NPCs/Monsters present.
4. **Summary**: Provide a 1-sentence summary of what this room is.

Output Format:
- **Location ID**: {area_id}
- **Summary**: ...
- **Exits**: [List of connected areas or directions]
- **Features**: [List of key elements]
"""

    @staticmethod
    def _prompt_for_section(context: str, children_text: str) -> str:
        return f"""
You are a D&D Adventure Assistant. Summarize the following SECTION.
This section may contain sub-sections (children). Synthesize the information.

{context}

Child Summaries:
{children_text}

Task:
1. Provide a high-level summary of this section's narrative or rule purpose.
2. Connect the child summaries into a cohesive flow.
3. Identify the main goal for the players in this section.
"""

    @staticmethod
    def _prompt_for_entries(context: str, children_text: str) -> str:
        return f"""
You are a D&D Rules Expert. Extract key details from this ENTRY.
Entries usually describe specific rooms, mechanics, or creature behaviors.

{context}

Child Summaries:
{children_text}

Task:
1. Summarize the immediate situation or description.
2. EXTRACT KEY MECHANICS: Look for DCs (e.g., DC 15), damage (e.g., 1d6), or conditions.
3. If this is a room description, list the key interactive objects.
"""

    @staticmethod
    def _prompt_for_inset(context: str, inset_type: str) -> str:
        role = "Player Description (Read Aloud)" if "Readaloud" in inset_type else "DM Secret Note"
        return f"""
You are analyzing a specific text block type: {role}.

{context}

Task:
1. If this is 'Readaloud': What is the mood/atmosphere? What visual/auditory clues do players get?
2. If this is 'DM Note': What is the hidden truth or rule the DM must know?
3. Summarize in 1-2 sentences.
"""

    @staticmethod
    def _prompt_for_image(node: Dict) -> str:
        # 提取图片元数据
        title = node.get("title", node.get("name", "Untitled Image"))
        img_type = node.get("imageType", "illustration") # map, mapPlayer, etc.
        credit = node.get("credit", "Unknown Artist")
        
        return f"""
You are analyzing metadata for a visual element in a D&D adventure.

Image Title: {title}
Image Type: {img_type}
Artist: {credit}

Task:
1. Identify the purpose of this image based on its title and type.
2. If it is a 'map' or 'mapPlayer', explicitly state that this node represents the spatial layout of the area.
3. If it is an illustration, describe it as a visual reference for the DM or players.
4. Do not hallucinate visual details not present in the metadata.
"""

    @staticmethod
    def _prompt_for_gallery(node: Dict) -> str:
        # 提取画廊中的图片列表
        images = node.get("images", [])
        img_summaries = []
        for img in images:
            t = img.get("title", "Untitled")
            it = img.get("imageType", "illustration")
            img_summaries.append(f"- {t} (Type: {it})")
        
        images_list_str = "\n".join(img_summaries)

        return f"""
You are analyzing a GALLERY node which contains a collection of images.

Images in this gallery:
{images_list_str}

Task:
1. Summarize what this collection represents (e.g., "A collection of maps for Stormwreck Isle").
2. Note if there are player-facing versions of maps (indicated by 'mapPlayer' or similar titles).
3. Treat this node as a container for visual aids.
"""

    @staticmethod
    def _prompt_for_list(context: str, node: Dict) -> str:
        # 处理列表项
        items = node.get("items", [])
        # 如果 items 是字符串列表
        if items and isinstance(items[0], str):
            items_text = "\n".join([f"- {item}" for item in items])
        # 如果 items 是对象列表 (如 adventure-dosi.json 中的情况)
        elif items and isinstance(items[0], dict):
             items_text = "\n".join([f"- {item.get('name', '')}: {item.get('entry', '')}" for item in items])
        else:
            items_text = "Empty list"

        return f"""
You are summarizing a LIST of items/options in a D&D module.

Context:
{context}

List Items:
{items_text}

Task:
1. Summarize the common theme of these items.
2. If they are choices (e.g., character hooks), list the options briefly.
3. If they are rules/steps, summarize the procedure.
"""

    @staticmethod
    def _prompt_default(context: str, children_text: str) -> str:
        return f"""
Summarize the following D&D content node.

{context}

Child Summaries:
{children_text}

Task:
Summarize the key information concisely.
"""