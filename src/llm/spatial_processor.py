import json
import re
from typing import List, Dict, Any
from openai import OpenAI
from src.llm.prompt_factory import PromptFactory
from src.utils.logger import logger

class SpatialTopologyProcessor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def process(self, shadow_tree: List[Dict]) -> Dict:
        full_graph = {"nodes": [], "edges": []}
        for root in shadow_tree:
            # 1. 递归生成文本总结 (不再预先过滤)
            self._recursive_summarize(root)
            
            # 2. 从根节点（或章节节点）提取图谱
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO":
                logger.info(f"Extracting Graph from summary of: {root['title']}")
                chapter_graph = self._extract_graph_from_text(root["spatial_summary"])
                full_graph["nodes"].extend(chapter_graph.get("nodes", []))
                full_graph["edges"].extend(chapter_graph.get("edges", []))

        return full_graph

    def _recursive_summarize(self, node: Dict) -> str:
        """
        递归生成空间总结文本 (Bottom-Up)
        """
        title = node.get("title", "Untitled")
        
        # 1. 先递归处理子节点
        valid_child_summaries = []
        if "children" in node:
            for child in node["children"]:
                s = self._recursive_summarize(child)
                # 只有当子节点有有效的空间信息时，才收集它的摘要
                if s and s != "NO_SPATIAL_INFO":
                    valid_child_summaries.append(s)
        
        children_text = "\n".join(valid_child_summaries)

        # 2. 准备调用 LLM
        # 即使没有 content，如果它有包含空间信息的子节点（比如 Chapter 1 包含 A1, A2），
        # 它本身也代表了一个“区域容器”，所以也需要总结。
        has_content = bool(node.get("content") or node.get("links"))
        has_valid_children = bool(valid_child_summaries)

        if has_content or has_valid_children:
            # 不再检查正则，直接生成 Prompt
            prompt = PromptFactory.create_spatial_summary_prompt(
                node, 
                children_text
            )
            
            # 调用 LLM
            summary = self._call_llm(prompt)
            
            # 3. 处理结果
            if "NO_SPATIAL_INFO" in summary:
                node["spatial_summary"] = "NO_SPATIAL_INFO"
                return "NO_SPATIAL_INFO"
            else:
                # 加上标题前缀，方便父级引用
                final_summary = f"[{title}]: {summary}"
                node["spatial_summary"] = final_summary
                return final_summary
        
        return "NO_SPATIAL_INFO"

    def _extract_graph_from_text(self, text: str) -> Dict:
        """
        最后一步：从汇总文本中提取 JSON
        """
        prompt = PromptFactory.create_graph_extraction_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            return {"nodes": [], "edges": []}

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""