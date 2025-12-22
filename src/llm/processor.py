import os
import json
import time
from typing import Dict, List, Any
from openai import OpenAI
from src.llm.prompt_factory import PromptFactory
from src.utils.logger import logger

class LLMProcessor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.request_count = 0

    def process_tree(self, shadow_tree: List[Dict]) -> List[Dict]:
        """
        处理整个影子树列表（通常是根节点列表）。
        """
        processed_tree = []
        total_nodes = len(shadow_tree)
        
        for index, root_node in enumerate(shadow_tree):
            logger.info(f"Processing root node {index + 1}/{total_nodes}: {root_node.get('title')}")
            processed_node = self._process_node_recursive(root_node)
            processed_tree.append(processed_node)
            
        return processed_tree

    def _process_node_recursive(self, node: Dict) -> Dict:
        """
        递归处理单个节点（后序遍历）。
        """
        node_id = node.get("id", "unknown")
        title = node.get("title", "Untitled")
        
        # 1. 递归处理子节点 (Bottom-Up)
        child_summaries = []
        if "children" in node and node["children"]:
            logger.info(f"Descending into children of: {title}")
            for child in node["children"]:
                processed_child = self._process_node_recursive(child)
                
                # 收集子节点的摘要，用于父节点 Prompt
                child_summary = processed_child.get("llm_summary", "")
                if child_summary:
                    child_summaries.append(f"[{processed_child.get('title')}]: {child_summary}")
        
        # 2. 生成当前节点的 Prompt
        # 只有当节点有内容、有链接或者有子节点摘要时才需要总结
        if node.get("content") or node.get("links") or child_summaries or node.get("type") in ["image", "gallery"]:
            prompt = PromptFactory.create_prompt(node, child_summaries)
            
            # 3. 调用 LLM
            logger.info(f"Summarizing node: {title} (ID: {node_id})")
            summary = self._call_openai(prompt)
            node["llm_summary"] = summary
        else:
            logger.warning(f"Skipping empty node: {title}")
            node["llm_summary"] = "No content to summarize."

        return node

    def _call_openai(self, prompt: str) -> str:
        """
        调用 OpenAI API，包含简单的重试逻辑。
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful D&D assistant specialized in summarizing adventure modules."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3, # 较低的温度以获得更确定的事实性总结
                )
                self.request_count += 1
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                logger.error(f"OpenAI API Error (Attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 * (attempt + 1)) # 指数退避
        
        return "Error: Failed to generate summary after retries."