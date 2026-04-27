import json
import re
import asyncio
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from src.llm.context_window import build_section_context, clamp_text, split_batches
from src.llm.prompt_factory import PromptFactory
from src.utils.logger import logger

class SpatialTopologyProcessor:
    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 100,
        context_window_chars: int = 24000,
        location_batch_size: int = 150,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url) # Keep synchronous client as backup
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent) # Limit concurrency
        self.context_window_chars = context_window_chars
        self.location_batch_size = location_batch_size

    def process(self, shadow_tree: List[Dict], skip_summary: bool = False) -> Dict:
        """Synchronous entry point that runs the async processing loop.

        Args:
            shadow_tree: The shadow tree structure representing document sections.
            skip_summary: Whether to skip the summarization step (if already done).

        Returns:
            A dictionary representing the full location knowledge graph associated with the tree.
        """
        return asyncio.run(self._process_async(shadow_tree, skip_summary))

    async def _process_async(self, shadow_tree: List[Dict], skip_summary: bool) -> Dict:
        full_graph = {"nodes": [], "edges": []}
        
        # 1. Parallel Recursive Summarization
        # Process top-level nodes (usually chapters) in parallel as well.
        if not skip_summary:
            tasks = [self._recursive_summarize_async(root) for root in shadow_tree]
            await asyncio.gather(*tasks)

        # 2. Concatenate all chapter summaries
        chapter_summaries = []
        for root in shadow_tree:
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO":
                chapter_summaries.append(root["spatial_summary"])

        if not chapter_summaries:
            return {"nodes": [], "edges": []}

        # 3. Extract graphs per chapter
        logger.info(f"Extracting graphs from {len(chapter_summaries)} chapters independently...")
        extraction_tasks = [self._extract_graph_from_text_async(summary) for summary in chapter_summaries]
        sub_graphs = await asyncio.gather(*extraction_tasks)

        full_graph = await self._merge_graphs(sub_graphs)

        return full_graph

    async def _recursive_summarize_async(self, node: Dict) -> str:
        """Recursively generates spatial summaries (Bottom-Up) - Async version.

        Args:
            node: The current node in the shadow tree.

        Returns:
            The summary string for the current node.
        """
        title = node.get("title", "Untitled")
        
        # 1. Process all children in parallel
        valid_child_summaries = []
        if "children" in node and node["children"]:
            # Create tasks for all children
            tasks = [self._recursive_summarize_async(child) for child in node["children"]]
            # Wait for all children to complete
            results = await asyncio.gather(*tasks)
            
            for s in results:
                if s and s != "NO_SPATIAL_INFO":
                    valid_child_summaries.append(s)
        
        children_text = "\n".join(valid_child_summaries)

        # 2. Prepare for LLM call
        has_content = bool(node.get("content") or node.get("links"))
        has_valid_children = bool(valid_child_summaries)

        if has_content or has_valid_children:
            prompt = PromptFactory.create_spatial_summary_prompt(
                node,
                clamp_text(children_text, self.context_window_chars // 2),
            )
            
            # Async LLM call
            summary = await self._call_llm_async(prompt)
            
            # 3. Handle result
            if "NO_SPATIAL_INFO" in summary:
                node["spatial_summary"] = "NO_SPATIAL_INFO"
                return "NO_SPATIAL_INFO"
            else:
                final_summary = f"[{title}]: {summary}"
                node["spatial_summary"] = final_summary
                return final_summary
        
        return "NO_SPATIAL_INFO"

    async def _extract_graph_from_text_async(self, text: str) -> Dict:
        """Final step: Extract JSON from summary text - Async version.

        Args:
            text: The summarized spatial text.

        Returns:
            A dictionary containing the extracted graph (nodes and edges).
        """
        prompt = PromptFactory.create_graph_extraction_prompt(
            clamp_text(text, self.context_window_chars)
        )
        
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
        """Merges multiple sub-graphs using LLM for entity alignment.

        Args:
            sub_graphs: A list of graph dictionaries to merge.

        Returns:
            A single merged graph dictionary.
        """
        all_nodes = []
        all_edges = []
        
        # 1. Collect all raw data
        for g in sub_graphs:
            all_nodes.extend(g.get("nodes", []))
            all_edges.extend(g.get("edges", []))
            
        if not all_nodes:
            return {"nodes": [], "edges": []}

        # 2. Prepare node list for LLM analysis
        # Format: "- [id] label (type)"
        node_list_text = "\n".join([f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})" for n in all_nodes])
        
        # 3. Call LLM to generate mapping table
        logger.info("Resolving entity duplicates with LLM...")
        mapping = await self._resolve_entities_with_llm(node_list_text)
        
        # 4. Reconstruct graph
        final_nodes = {}
        final_edges = []
        seen_edges = set()

        # Apply mapping to nodes
        for node in all_nodes:
            original_id = node["id"]
            # Get canonical ID, default to original if no mapping exists
            canonical_id = mapping.get(original_id, original_id)
            
            # Update node ID
            node["id"] = canonical_id
            
            # Save node (simple overwrite strategy, keeping the last one encountered)
            # Optimization: Could keep the one with the longest label
            if canonical_id not in final_nodes:
                final_nodes[canonical_id] = node
            else:
                if len(node.get("label", "")) > len(final_nodes[canonical_id].get("label", "")):
                    final_nodes[canonical_id]["label"] = node["label"]

        # Apply mapping to edges
        for edge in all_edges:
            source = mapping.get(edge["source"], edge["source"])
            target = mapping.get(edge["target"], edge["target"])
            
            # Ignore self-loops (unless special logic applies)
            if source == target:
                continue
                
            relationship = edge.get("relationship") or edge.get("relation") or ""
            edge_key = f"{source}|{relationship}|{target}"
            
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
        """Request LLM to identify duplicate entities and return an ID mapping.

        Args:
            node_list_text: Text representation of the list of nodes.

        Returns:
            A dictionary mapping duplicate IDs to canonical IDs.
        """
        prompt = PromptFactory.create_entity_resolution_prompt(
            clamp_text(node_list_text, self.context_window_chars)
        )
        
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


class SectionLocationMapper:
    """Specific class for mapping Shadow Tree sections to known nodes in the Location Graph."""

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 100,
        context_window_chars: int = 24000,
        location_batch_size: int = 150,
    ):
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.context_window_chars = context_window_chars
        self.location_batch_size = location_batch_size

    def process(self, shadow_tree: List[Dict], location_graph: Dict) -> Dict[str, str]:
        """Synchronous entry point.

        Args:
            shadow_tree: The shadow tree structure.
            location_graph: The existing location graph to map against.

        Returns:
            A dictionary mapping section IDs to location IDs. { "section_id": "location_id" }
        """
        return asyncio.run(self._process_async(shadow_tree, location_graph))

    async def _process_async(self, shadow_tree: List[Dict], location_graph: Dict) -> Dict[str, str]:
        location_ids = [n["id"] for n in location_graph.get("nodes", [])]
        if not location_ids:
            logger.warning("No locations found in graph, skipping mapping.")
            return {}

        # 1. Collect sections that need mapping
        sections = self._collect_sections_info(shadow_tree)
        logger.info(f"Mapping {len(sections)} sections to {len(location_ids)} locations...")

        # 2. Execute mapping concurrently
        tasks = [self._map_section_to_locations(s, location_ids) for s in sections]
        results = await asyncio.gather(*tasks)

        # 3. Organize results
        mapping = {}
        for sec_id, loc_ids in results:
            valid_ids = [lid for lid in loc_ids if lid in location_ids]
            if valid_ids:
                mapping[sec_id] = valid_ids
        
        logger.info(f"Successfully mapped {len(mapping)} sections.")
        return mapping

    async def _map_section_to_locations(self, section_ctx: Dict, location_ids: List[str]) -> tuple:
        content_budget = max(2000, self.context_window_chars // 2)
        context = build_section_context(section_ctx, content_budget)

        batches = split_batches(location_ids, self.location_batch_size)
        all_loc_ids = []
        for batch in batches:
            batch_loc_ids = await self._map_section_batch(context, section_ctx["id"], batch)
            all_loc_ids.extend(batch_loc_ids)

        return section_ctx["id"], all_loc_ids

    async def _map_section_batch(
        self,
        section_context: str,
        section_id: str,
        location_ids: List[str],
    ) -> List[str]:
        loc_list_str = ", ".join(location_ids)
        prompt = PromptFactory.create_section_mapping_prompt(section_context, loc_list_str)
        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                data = json.loads(response.choices[0].message.content)
                
                # Compatibility handling
                raw_result = data.get("location_ids") or data.get("location_id")
                
                if isinstance(raw_result, list):
                    loc_ids = raw_result
                elif isinstance(raw_result, str):
                    loc_ids = [raw_result]
                else:
                    loc_ids = []

                return loc_ids
        except Exception as e:
            logger.error(f"Mapping failed for section {section_id}: {e}")
            return []

    def _collect_sections_info(
        self,
        nodes: List[Dict],
        parent_title: str = "",
        path: List[str] | None = None,
        parent_spatial_summary: str = "",
    ) -> List[Dict]:
        """Recursively collects all Sections containing content or links, attached with parent title info.

        Args:
            nodes: List of nodes to process.
            parent_title: Title of the parent node.

        Returns:
            A list of dictionaries containing section context.
        """
        collected = []
        path = path or []
        sibling_titles = [node.get("title", "Untitled") for node in nodes]
        for node in nodes:
            current_title = node.get("title", "Untitled")

            children = node.get("children", [])
            child_titles = [child.get("title", "Untitled") for child in children]
            current_path = [*path, current_title]
            
            context_obj = {
                "id": node["id"],
                "title": current_title,
                "parent_title": parent_title,
                "content": node.get("content", ""),
                "child_titles": child_titles,
                "sibling_titles": [title for title in sibling_titles if title != current_title],
                "path": current_path,
                "parent_spatial_summary": parent_spatial_summary,
                "spatial_summary": node.get("spatial_summary", ""),
            }
            
            # Only map nodes that have content
            if context_obj["content"]:
                collected.append(context_obj)
            
            if "children" in node:
                collected.extend(
                    self._collect_sections_info(
                        node["children"],
                        current_title,
                        current_path,
                        node.get("spatial_summary", parent_spatial_summary),
                    )
                )
        return collected
