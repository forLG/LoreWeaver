import json
import re
import asyncio
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from llm.prompt_factory import PromptFactory
from utils.logger import logger

class SpatialTopologyProcessor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat", max_concurrent: int = 100, use_multi_pass: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url) # 保留同步客户端备用
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent) # 限制并发数
        self.use_multi_pass = use_multi_pass  # Enable multi-pass extraction for smaller models

    def process(self, shadow_tree: List[Dict], skip_summary: bool = False) -> Dict:
        """
        同步入口，内部运行异步循环
        """
        return asyncio.run(self._process_async(shadow_tree, skip_summary))

    async def _process_async(self, shadow_tree: List[Dict], skip_summary: bool) -> Dict:
        # Route to multi-pass mode if enabled (for smaller models)
        if self.use_multi_pass:
            logger.info("Using multi-pass extraction mode (optimized for smaller models)")
            return await self._process_multi_pass_async(shadow_tree, skip_summary)

        # Standard single-pass mode (for larger models)
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
                
            edge_key = f"{source}|{edge.get('relation', 'related_to')}|{target}"
            
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

    # ========================================================================
    # Multi-Pass Extraction Methods (for smaller models like qwen3-8b)
    # ========================================================================

    async def _process_multi_pass_async(self, shadow_tree: List[Dict], skip_summary: bool) -> Dict:
        """
        Multi-pass extraction mode for smaller models.
        Breaks down the extraction into 4 passes to reduce cognitive load per pass.
        """
        # 1. Generate spatial summaries (same as standard mode)
        if not skip_summary:
            tasks = [self._recursive_summarize_async(root) for root in shadow_tree]
            await asyncio.gather(*tasks)

        # 2. Collect all chapter summaries
        chapter_summaries = []
        for root in shadow_tree:
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO":
                chapter_summaries.append(root["spatial_summary"])

        if not chapter_summaries:
            return {"nodes": [], "edges": []}

        # Combine all summaries for processing
        combined_summary = "\n\n".join(chapter_summaries)

        logger.info("=== Multi-Pass Extraction Starting ===")

        # Pass 1: Extract top-level hierarchy
        logger.info("Pass 1: Extracting top-level hierarchy (World, Region, Island)...")
        top_level_graph = await self._extract_top_level_async(combined_summary)

        # Pass 2: Extract sub-locations for each top-level region
        logger.info("Pass 2: Extracting sub-locations for each region...")
        detailed_graph = await self._extract_sub_locations_async(
            top_level_graph,
            combined_summary
        )

        # Pass 3: Extract additional relationships
        logger.info("Pass 3: Extracting additional relationships...")
        final_graph = await self._extract_relationships_async(
            detailed_graph,
            combined_summary
        )

        # Pass 4: Verification and refinement
        logger.info("Pass 4: Verifying and refining...")
        final_graph = await self._verify_and_refine_async(
            final_graph,
            combined_summary
        )

        logger.info(f"=== Multi-Pass Complete: {len(final_graph['nodes'])} nodes, {len(final_graph['edges'])} edges ===")

        return final_graph

    async def _extract_top_level_async(self, combined_summary: str) -> Dict:
        """Pass 1: Extract only top-level locations (World, Region, Island, City)"""
        prompt = PromptFactory.create_top_level_extraction_prompt(combined_summary)

        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=8192  # Pass 1: Top-level extraction can be large
                )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Top-level extraction failed: {e}")
            return {"nodes": [], "edges": []}

    async def _extract_sub_locations_async(self, top_level_graph: Dict, combined_summary: str) -> Dict:
        """Pass 2: Extract sub-locations for each top-level region"""
        # Find regions/islands that need sub-location extraction
        parent_candidates = [
            node for node in top_level_graph.get("nodes", [])
            if node.get("type") in ["Island", "Region", "City", "Building", "Cave System"]
        ]

        if not parent_candidates:
            logger.info("No parent candidates found for sub-location extraction")
            return top_level_graph

        # Prepare existing nodes text for context
        existing_nodes_text = "\n".join([
            f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})"
            for n in top_level_graph.get("nodes", [])
        ])

        # Extract sub-locations for each parent (parallel)
        sub_location_tasks = [
            self._extract_sub_locations_for_parent_async(
                parent["id"],
                parent["label"],
                combined_summary,
                existing_nodes_text
            )
            for parent in parent_candidates
        ]

        sub_graphs = await asyncio.gather(*sub_location_tasks)

        # Merge all sub-graphs
        result_graph = {
            "nodes": top_level_graph.get("nodes", []).copy(),
            "edges": top_level_graph.get("edges", []).copy()
        }

        for sub_graph in sub_graphs:
            result_graph["nodes"].extend(sub_graph.get("nodes", []))
            result_graph["edges"].extend(sub_graph.get("edges", []))

        logger.info(f"  Sub-location extraction: {len(result_graph['nodes'])} nodes, {len(result_graph['edges'])} edges")

        return result_graph

    async def _extract_sub_locations_for_parent_async(
        self, parent_id: str, parent_label: str, combined_summary: str, existing_nodes: str
    ) -> Dict:
        """Extract sub-locations for a single parent location"""
        prompt = PromptFactory.create_sub_location_extraction_prompt(
            parent_id, parent_label, combined_summary, existing_nodes
        )

        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=8192  # Pass 2: Sub-location extraction can be large
                )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Sub-location extraction for {parent_label} failed: {e}")
            return {"nodes": [], "edges": []}

    async def _extract_relationships_async(self, graph: Dict, combined_summary: str) -> Dict:
        """Pass 3: Extract additional relationships between locations"""
        # Prepare nodes text
        nodes_text = "\n".join([
            f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})"
            for n in graph.get("nodes", [])
        ])

        prompt = PromptFactory.create_relationship_extraction_prompt(nodes_text, combined_summary)

        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=8192  # Pass 3: Relationship extraction can be large
                )
            result = json.loads(response.choices[0].message.content)

            # Merge new edges
            existing_edges = graph.get("edges", [])
            new_edges = result.get("edges", [])

            # Deduplicate edges
            seen_edges = {f"{e['source']}|{e.get('relation', 'related_to')}|{e['target']}" for e in existing_edges}
            for edge in new_edges:
                edge_key = f"{edge['source']}|{edge.get('relation', 'related_to')}|{edge['target']}"
                if edge_key not in seen_edges:
                    existing_edges.append(edge)
                    seen_edges.add(edge_key)

            graph["edges"] = existing_edges
            logger.info(f"  Relationship extraction: added {len(new_edges)} edges")

            return graph
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return graph

    async def _verify_and_refine_async(self, graph: Dict, combined_summary: str) -> Dict:
        """Pass 4: Verification and refinement"""
        # Prepare graph text
        nodes_text = "\n".join([
            f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})"
            for n in graph.get("nodes", [])
        ])
        edges_text = "\n".join([
            f"- {e.get('source')} -> {e.get('target')} ({e.get('relation')})"
            for e in graph.get("edges", [])
        ])
        graph_text = f"NODES:\n{nodes_text}\n\nEDGES:\n{edges_text}"

        prompt = PromptFactory.create_multi_pass_verification_prompt(graph_text, combined_summary)

        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=8192  # Pass 4: Verification can be large
                )
            result = json.loads(response.choices[0].message.content)

            if result.get("revised_edges"):
                # Merge revised edges
                for edge in result.get("revised_edges", []):
                    edge_key = f"{edge['source']}|{edge.get('relation', 'related_to')}|{edge['target']}"
                    # Check if edge exists
                    exists = any(
                        e.get("source") == edge["source"] and
                        e.get("target") == edge["target"] and
                        e.get("relation") == edge.get("relation")
                        for e in graph["edges"]
                    )
                    if not exists:
                        graph["edges"].append(edge)

            if result.get("issues"):
                logger.info("  Verification issues found:")
                for issue in result.get("issues", []):
                    logger.info(f"    - {issue}")

            return graph
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return graph


class SectionLocationMapper:
    """
    专门负责将 Shadow Tree 的章节映射到已知的 Location Graph 节点上
    """
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat", max_concurrent: int = 100):
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def process(self, shadow_tree: List[Dict], location_graph: Dict) -> Dict[str, str]:
        """
        同步入口
        :return: { "section_id": "location_id" }
        """
        return asyncio.run(self._process_async(shadow_tree, location_graph))

    async def _process_async(self, shadow_tree: List[Dict], location_graph: Dict) -> Dict[str, str]:
        location_ids = [n["id"] for n in location_graph.get("nodes", [])]
        if not location_ids:
            logger.warning("No locations found in graph, skipping mapping.")
            return {}

        # 1. 收集需要映射的 Section
        sections = self._collect_sections_info(shadow_tree)
        logger.info(f"Mapping {len(sections)} sections to {len(location_ids)} locations...")

        # 2. 并发执行映射
        tasks = [self._map_section_to_locations(s, location_ids) for s in sections]
        results = await asyncio.gather(*tasks)

        # 3. 整理结果
        mapping = {}
        for sec_id, loc_ids in results:
            # Handle None from failed JSON parsing
            if loc_ids is None:
                logger.warning(f"Skipping section {sec_id} due to mapping error")
                continue
            valid_ids = [lid for lid in loc_ids if lid in location_ids]
            if valid_ids:
                mapping[sec_id] = valid_ids

        logger.info(f"Successfully mapped {len(mapping)} sections.")
        return mapping

    async def _map_section_to_locations(self, section_ctx: Dict, location_ids: List[str]) -> tuple:
        child_titles_str = ", ".join(section_ctx.get('child_titles', [])) or "None"

        # Truncate content to avoid overwhelming the model
        content = section_ctx.get('content', '')
        if len(content) > 3000:  # Limit content length for smaller models
            content = content[:3000] + "... [truncated]"

        context = (
            f"ID: {section_ctx['id']}\n"
            f"Parent Title: {section_ctx['parent_title']}\n"
            f"Title: {section_ctx['title']}\n"
            f"Content: {content}\n"
            f"Child Titles: {child_titles_str}"
        )

        loc_list_str = ", ".join(location_ids)
        prompt = PromptFactory.create_section_mapping_prompt(context, loc_list_str)

        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=4096  # Increased for very long location lists
                )

                raw_content = response.choices[0].message.content

                # Log raw response for debugging (only on error)
                try:
                    data = json.loads(raw_content)
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parse error for section {section_ctx['id']}: {je}")
                    logger.error(f"Raw response (first 500 chars): {raw_content[:500]}...")
                    return section_ctx["id"], None

                # 兼容性处理
                raw_result = data.get("location_ids") or data.get("location_id")

                if isinstance(raw_result, list):
                    loc_ids = raw_result
                elif isinstance(raw_result, str):
                    loc_ids = [raw_result]
                else:
                    loc_ids = []

                return section_ctx["id"], loc_ids
        except Exception as e:
            logger.error(f"Mapping failed for section {section_ctx['id']}: {e}")
            return section_ctx["id"], None

    def _collect_sections_info(self, nodes: List[Dict], parent_title: str = "") -> List[Dict]:
        """
        递归收集所有包含内容或链接的 Section，并附带父级标题信息
        """
        collected = []
        for node in nodes:
            current_title = node.get("title", "Untitled")

            children = node.get("children", [])
            child_titles = [child.get("title", "Untitled") for child in children]
            
            context_obj = {
                "id": node["id"],
                "title": current_title,
                "parent_title": parent_title,
                "content": node.get("content", ""),
                "child_titles": child_titles
            }
            
            # 只有有内容的节点才值得映射
            if context_obj["content"]:
                collected.append(context_obj)
            
            if "children" in node:
                collected.extend(self._collect_sections_info(node["children"], current_title))
        return collected