import json
from typing import Dict, List, Any, Optional
from src.utils.link_processor import LinkProcessor
from src.utils.logger import logger

class ShadowNode:
    def __init__(self, id: str, title: str, node_type: str):
        self.id = id
        self.title = title
        self.type = node_type
        self.text_content: List[str] = [] # 累积当前节点的文本段落
        self.links: List[Dict] = []       # 累积当前节点的链接
        self.children: List['ShadowNode'] = []

    def add_text(self, text: str):
        if not text:
            return
        clean_text, links = LinkProcessor.parse_and_clean(text)
        self.text_content.append(clean_text)
        self.links.extend(links)

    def to_dict(self):
        # 在输出前对累积的所有链接进行最终去重
        # 由于 `add_text` 会被多次调用，直接在那里去重会有
        # 较大的时间开销
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
            "content": "\n".join(self.text_content), # 合并文本段落
            "links": unique_links,
            "children": [child.to_dict() for child in self.children]
        }

class ShadowTreeBuilder:
    def build(self, data: List[Dict]) -> List[Dict]:
        """构建整个影子树"""
        roots = []
        for entry in data:
            node = self._process_entry(entry)
            if node:
                roots.append(node.to_dict())
        return roots

    def _process_entry(self, entry: Any) -> Optional[ShadowNode]:
        """
        递归处理入口。
        返回 ShadowNode 表示这是一个独立的节点（如 Section）。
        返回 None 表示这是一个内容块（如 String, List），其内容已被合并到父级（逻辑需在调用方处理，或者这里我们简化逻辑：只处理结构性节点）。
        
        注意：为了简化递归，我们采用一种策略：
        - 如果遇到 Section/Inset/Entries 类型 -> 创建新节点，递归处理子项。
        - 如果遇到 String/List/Image -> 它们属于“当前上下文”，不创建独立节点。
        
        但在递归函数中，我们很难直接“返回给父级文本”。
        所以我们将逻辑拆分为：_create_node_from_structure 和 _extract_content_from_structure。
        """
        
        # 1. 结构性节点：创建新的 ShadowNode
        if isinstance(entry, dict) and entry.get("type") in ["section", "entries", "inset", "insetReadaloud"]:
            node_id = entry.get("id", "")
            title = entry.get("name", "Untitled Section")
            node_type = entry.get("type")
            
            node = ShadowNode(node_id, title, node_type)
            
            # 处理该节点下的 entries
            if "entries" in entry:
                for child_entry in entry["entries"]:
                    # 递归检查子项
                    if self._is_structural_node(child_entry):
                        # 如果子项也是结构性节点（如 Section 嵌套 Section），添加到 children
                        child_node = self._process_entry(child_entry)
                        if child_node:
                            node.children.append(child_node)
                    else:
                        # 如果子项是内容（如 String, List），提取文本和链接到当前节点
                        self._extract_content(child_entry, node)
            
            return node
            
        return None

    def _is_structural_node(self, entry: Any) -> bool:
        """判断是否应该成为独立的子节点"""
        if isinstance(entry, dict):
            # 这些类型通常包含大量子内容，适合作为独立节点
            return entry.get("type") in ["section", "entries", "inset", "insetReadaloud"]
        return False

    def _extract_content(self, entry: Any, current_node: ShadowNode):
        """提取非结构化节点的内容，合并到当前节点中"""
        
        # Case 1: 纯字符串
        if isinstance(entry, str):
            current_node.add_text(entry)
            
        # Case 2: 字典类型 (List, Image, Table, Quote 等)
        elif isinstance(entry, dict):
            entry_type = entry.get("type")
            
            if entry_type == "list":
                # 处理列表项
                for item in entry.get("items", []):
                    if isinstance(item, str):
                        current_node.add_text(f"- {item}")
                    elif isinstance(item, dict) and "entry" in item:
                        # 处理带标题的列表项 (name: entry)
                        name = item.get("name", "")
                        text = item.get("entry", "")
                        full_text = f"- **{name}**: {text}" if name else f"- {text}"
                        current_node.add_text(full_text)
            
            elif entry_type == "table":
                # 简单处理表格，提取 caption 和 row 内容
                if "caption" in entry:
                    current_node.add_text(f"**Table: {entry['caption']}**")
                for row in entry.get("rows", []):
                    # row 可能是字符串列表
                    row_text = " | ".join([str(cell) for cell in row])
                    current_node.add_text(row_text)

            elif entry_type == "image":
                # 仅处理具有 title 的图片（通常是地图或重要图示）
                title = entry.get("title", "")
                image_type = entry.get("imageType", "")
                if title:
                    if image_type:
                        current_node.add_text(f"[Image ({image_type}): {title}]")
                    else: 
                        current_node.add_text(f"[Image: {title}]")

            elif entry_type == "gallery":
                # 处理画廊，有有效图片时才处理
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
                # 引用块
                for line in entry.get("entries", []):
                    if isinstance(line, str):
                        current_node.add_text(f"> {line}")

            else:
                # 未知类型，记录警告
                logger.warning(f"Unknown entry type encountered: '{entry_type}' in node '{current_node.title}' (ID: {current_node.id})")