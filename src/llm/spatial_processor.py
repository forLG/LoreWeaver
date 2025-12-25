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

    def process(self, shadow_tree: List[Dict], skip_summary: bool = False) -> Dict:
        """
        同步入口，内部运行异步循环
        """
        return asyncio.run(self._process_async(shadow_tree, skip_summary))

    async def _process_async(self, shadow_tree: List[Dict], skip_summary: bool) -> Dict:
        full_graph = {"nodes": [], "edges": []}
        
        # 1. 并行递归生成文本总结
        # 对顶层节点（通常是章节）也进行并行处理
        if not skip_summary:
            tasks = [self._recursive_summarize_async(root) for root in shadow_tree]
            await asyncio.gather(*tasks)

        # 2. 将所有章节的总结拼接
        chapter_summaries = []
        for root in shadow_tree:
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO":
                chapter_summaries.append(root["spatial_summary"])

        if not chapter_summaries:
            return {"nodes": [], "edges": []}

        # 3. 分章提取图谱
        logger.info(f"Extracting graphs from {len(chapter_summaries)} chapters independently...")
        extraction_tasks = [self._extract_graph_from_text_async(summary) for summary in chapter_summaries]
        sub_graphs = await asyncio.gather(*extraction_tasks)

        full_graph = await self._merge_graphs(sub_graphs)

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

    async def _merge_graphs(self, sub_graphs: List[Dict]) -> Dict:
        """
        使用 LLM 辅助进行实体对齐和图谱合并
        """
        all_nodes = []
        all_edges = []
        
        # 1. 收集所有原始数据
        for g in sub_graphs:
            all_nodes.extend(g.get("nodes", []))
            all_edges.extend(g.get("edges", []))
            
        if not all_nodes:
            return {"nodes": [], "edges": []}

        # 2. 准备节点列表供 LLM 分析
        # 格式: "- [id] label (type)"
        node_list_text = "\n".join([f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})" for n in all_nodes])
        
        # 3. 调用 LLM 生成映射表
        logger.info("Resolving entity duplicates with LLM...")
        mapping = await self._resolve_entities_with_llm(node_list_text)
        
        # 4. 重构图谱
        final_nodes = {}
        final_edges = []
        seen_edges = set()

        # 应用映射表处理节点
        for node in all_nodes:
            original_id = node["id"]
            # 获取映射后的 ID，如果没有映射则保持原样
            canonical_id = mapping.get(original_id, original_id)
            
            # 更新节点 ID
            node["id"] = canonical_id
            
            # 保存节点 (简单的覆盖策略，保留最后一个遇到的)
            # 优化：可以保留 label 最长的那个
            if canonical_id not in final_nodes:
                final_nodes[canonical_id] = node
            else:
                if len(node.get("label", "")) > len(final_nodes[canonical_id].get("label", "")):
                    final_nodes[canonical_id]["label"] = node["label"]

        # 应用映射表处理边
        for edge in all_edges:
            source = mapping.get(edge["source"], edge["source"])
            target = mapping.get(edge["target"], edge["target"])
            
            # 忽略自环 (除非是特殊的自环逻辑，一般图谱里不需要)
            if source == target:
                continue
                
            edge_key = f"{source}|{edge.get('relation')}|{target}"
            
            if edge_key not in seen_edges:
                new_edge = edge.copy()
                new_edge["source"] = source
                new_edge["target"] = target
                final_edges.append(new_edge)
                seen_edges.add(edge_key)

        return {
            "nodes": list(final_nodes.values()),
            "edges": final_edges
        }
    
    async def _resolve_entities_with_llm(self, node_list_text: str) -> Dict[str, str]:
        """
        请求 LLM 找出重复的实体并返回 ID 映射表
        """
        prompt = PromptFactory.create_entity_resolution_prompt(node_list_text)
        
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
            logger.error(f"Entity resolution failed: {e}")
            return {}

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