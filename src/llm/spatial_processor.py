import json
import re
import asyncio
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from src.llm.prompt_factory import PromptFactory
from src.utils.logger import logger

class SpatialTopologyProcessor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat", max_concurrent: int = 100):
        self.client = OpenAI(api_key=api_key, base_url=base_url) # 保留同步客户端备用
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent) # 限制并发数

    def process(self, shadow_tree: List[Dict]) -> Dict:
        """
        同步入口，内部运行异步循环
        """
        return asyncio.run(self._process_async(shadow_tree))

    async def _process_async(self, shadow_tree: List[Dict]) -> Dict:
        full_graph = {"nodes": [], "edges": []}
        
        # 1. 并行递归生成文本总结
        # 对顶层节点（通常是章节）也进行并行处理
        tasks = [self._recursive_summarize_async(root) for root in shadow_tree]
        await asyncio.gather(*tasks)

        # 2. 从根节点提取图谱
        for root in shadow_tree:
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO":
                logger.info(f"Extracting Graph from summary of: {root['title']}")
                # 这一步也可以异步，虽然通常调用次数较少
                chapter_graph = await self._extract_graph_from_text_async(root["spatial_summary"])
                full_graph["nodes"].extend(chapter_graph.get("nodes", []))
                full_graph["edges"].extend(chapter_graph.get("edges", []))

        return full_graph

    async def _recursive_summarize_async(self, node: Dict) -> str:
        """
        递归生成空间总结文本 (Bottom-Up) - 异步版
        """
        title = node.get("title", "Untitled")
        
        # 1. 并行处理所有子节点
        valid_child_summaries = []
        if "children" in node and node["children"]:
            # 创建所有子节点的任务列表
            tasks = [self._recursive_summarize_async(child) for child in node["children"]]
            # 等待所有子节点完成 (并发执行)
            results = await asyncio.gather(*tasks)
            
            for s in results:
                if s and s != "NO_SPATIAL_INFO":
                    valid_child_summaries.append(s)
        
        children_text = "\n".join(valid_child_summaries)

        # 2. 准备调用 LLM
        has_content = bool(node.get("content") or node.get("links"))
        has_valid_children = bool(valid_child_summaries)

        if has_content or has_valid_children:
            prompt = PromptFactory.create_spatial_summary_prompt(
                node, 
                children_text
            )
            
            # 异步调用 LLM
            summary = await self._call_llm_async(prompt)
            
            # 3. 处理结果
            if "NO_SPATIAL_INFO" in summary:
                node["spatial_summary"] = "NO_SPATIAL_INFO"
                return "NO_SPATIAL_INFO"
            else:
                final_summary = f"[{title}]: {summary}"
                node["spatial_summary"] = final_summary
                return final_summary
        
        return "NO_SPATIAL_INFO"

    async def _extract_graph_from_text_async(self, text: str) -> Dict:
        """
        最后一步：从汇总文本中提取 JSON - 异步版
        """
        prompt = PromptFactory.create_graph_extraction_prompt(text)
        
        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            return {"nodes": [], "edges": []}

    async def _call_llm_async(self, prompt: str) -> str:
        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""