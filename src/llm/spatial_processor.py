import asyncio

from openai import OpenAI

from llm.base_processor import BaseLLMProcessor
from llm.prompt_factory import PromptFactory
from utils.graph_utils import apply_entity_mapping, deduplicate_graph, format_graph_summary
from utils.logger import logger


class SpatialTopologyProcessor(BaseLLMProcessor):
    """
    Extract spatial topology graph from adventure content.

    Supports both single-pass and multi-pass extraction modes.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 100,
        use_multi_pass: bool = False
    ):
        super().__init__(api_key, base_url, model, max_concurrent)
        # Keep sync client for backward compatibility (unused in current code)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.use_multi_pass = use_multi_pass

    def process(self, shadow_tree: list[dict], skip_summary: bool = False) -> dict:
        """Synchronous entry point."""
        async def _run_and_cleanup():
            try:
                return await self._process_async(shadow_tree, skip_summary)
            finally:
                await self.close()

        return asyncio.run(_run_and_cleanup())

    async def _process_async(self, shadow_tree: list[dict], skip_summary: bool) -> dict:
        if self.use_multi_pass:
            logger.info("Using multi-pass extraction mode (optimized for smaller models)")
            return await self._process_multi_pass_async(shadow_tree, skip_summary)
        return await self._process_single_pass_async(shadow_tree, skip_summary)

    async def _process_single_pass_async(
        self,
        shadow_tree: list[dict],
        skip_summary: bool
    ) -> dict:
        """Standard single-pass extraction mode."""
        # 1. Generate spatial summaries
        if not skip_summary:
            tasks = [self._recursive_summarize_async(root) for root in shadow_tree]
            await asyncio.gather(*tasks)

        # 2. Collect chapter summaries
        chapter_summaries = [
            root["spatial_summary"]
            for root in shadow_tree
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO"
        ]

        if not chapter_summaries:
            return {"nodes": [], "edges": []}

        # 3. Extract graphs from each chapter
        logger.info(f"Extracting graphs from {len(chapter_summaries)} chapters independently...")
        extraction_tasks = [
            self._extract_graph_from_text_async(summary)
            for summary in chapter_summaries
        ]
        sub_graphs = await asyncio.gather(*extraction_tasks)

        return await self._merge_graphs_with_llm(sub_graphs)

    async def _recursive_summarize_async(self, node: dict) -> str:
        """Recursively generate spatial summary (bottom-up)."""
        title = node.get("title", "Untitled")

        # 1. Parallel process all children
        valid_child_summaries = []
        if node.get("children"):
            tasks = [self._recursive_summarize_async(child) for child in node["children"]]
            results = await asyncio.gather(*tasks)
            valid_child_summaries = [s for s in results if s and s != "NO_SPATIAL_INFO"]

        children_text = "\n".join(valid_child_summaries)

        # 2. Generate summary for this node
        has_content = bool(node.get("content") or node.get("links"))
        has_valid_children = bool(valid_child_summaries)

        if has_content or has_valid_children:
            prompt = PromptFactory.create_spatial_summary_prompt(node, children_text)
            summary = await self._call_llm_async(prompt)

            if "NO_SPATIAL_INFO" in summary:
                node["spatial_summary"] = "NO_SPATIAL_INFO"
                return "NO_SPATIAL_INFO"
            else:
                final_summary = f"[{title}]: {summary}"
                node["spatial_summary"] = final_summary
                return final_summary

        return "NO_SPATIAL_INFO"

    async def _extract_graph_from_text_async(self, text: str) -> dict:
        """Extract graph JSON from text summary."""
        prompt = PromptFactory.create_graph_extraction_prompt(text)
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context="Graph extraction"
        )
        return result if result else {"nodes": [], "edges": []}

    async def _merge_graphs_with_llm(self, sub_graphs: list[dict]) -> dict:
        """Merge graphs using LLM for entity resolution."""
        # Collect all nodes and edges
        all_nodes = []
        all_edges = []
        for g in sub_graphs:
            all_nodes.extend(g.get("nodes", []))
            all_edges.extend(g.get("edges", []))

        if not all_nodes:
            return {"nodes": [], "edges": []}

        # Use LLM to resolve duplicates
        node_list_text = "\n".join([
            f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})"
            for n in all_nodes
        ])

        logger.info("Resolving entity duplicates with LLM...")
        mapping = await self._resolve_entities_with_llm(node_list_text)

        return apply_entity_mapping({"nodes": all_nodes, "edges": all_edges}, mapping)

    async def _resolve_entities_with_llm(self, node_list_text: str) -> dict[str, str]:
        """Request LLM to find duplicate entities and return ID mapping."""
        prompt = PromptFactory.create_entity_resolution_prompt(node_list_text)
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context="Entity resolution"
        )
        return result if result else {}

    # ========================================================================
    # Multi-Pass Extraction (for smaller models)
    # ========================================================================

    async def _process_multi_pass_async(
        self,
        shadow_tree: list[dict],
        skip_summary: bool
    ) -> dict:
        """Multi-pass extraction for smaller models (qwen3-8b, etc.)."""
        # 1. Generate spatial summaries (same as single-pass)
        if not skip_summary:
            tasks = [self._recursive_summarize_async(root) for root in shadow_tree]
            await asyncio.gather(*tasks)

        # 2. Collect chapter summaries
        chapter_summaries = [
            root["spatial_summary"]
            for root in shadow_tree
            if root.get("spatial_summary") and root["spatial_summary"] != "NO_SPATIAL_INFO"
        ]

        if not chapter_summaries:
            return {"nodes": [], "edges": []}

        logger.info("=== Multi-Pass Extraction Starting ===")

        # 3. Choose strategy based on model capability
        is_small_model = any(x in self.model.lower() for x in ['qwen', 'llama', 'mistral', 'deepseek'])

        if is_small_model:
            # Small model: Process chapters incrementally to avoid context overflow
            return await self._process_small_model_multi_pass(chapter_summaries)
        else:
            # Large model: Can handle combined summaries
            combined_summary = "\n\n".join(chapter_summaries)
            return await self._process_large_model_multi_pass(combined_summary)

    async def _process_small_model_multi_pass(self, chapter_summaries: list[str]) -> dict:
        """
        Process multi-pass for small models with limited context.

        Strategy: Process each chapter separately, then merge and deduplicate results.
        """
        logger.info(f"Using small-model strategy for {len(chapter_summaries)} chapters")

        # Process each chapter independently
        all_graphs = []
        for i, summary in enumerate(chapter_summaries):
            logger.info(f"  Processing chapter {i+1}/{len(chapter_summaries)}...")
            graph = await self._extract_top_level_async(summary)
            all_graphs.append(graph)

        # Merge all chapter graphs
        merged = {"nodes": [], "edges": []}
        for g in all_graphs:
            merged["nodes"].extend(g.get("nodes", []))
            merged["edges"].extend(g.get("edges", []))

        logger.info(f"  Raw merge: {len(merged['nodes'])} nodes, {len(merged['edges'])} edges")

        # Deduplicate to resolve duplicates from independent chapter processing
        deduplicated = deduplicate_graph(merged)

        logger.info(f"  After deduplication: {len(deduplicated.nodes)} nodes, {len(deduplicated.edges)} edges")

        # Return as dict for JSON serialization
        return deduplicated.to_dict()

    async def _process_large_model_multi_pass(self, combined_summary: str) -> dict:
        """
        Process multi-pass for large models with ample context.

        Strategy: Run all passes on combined summary.
        """
        logger.info("Using large-model strategy with combined summary")

        # Pass 1: Top-level hierarchy
        logger.info("Pass 1: Extracting top-level hierarchy (World, Region, Island)...")
        top_level_graph = await self._extract_top_level_async(combined_summary)

        # Pass 2: Sub-locations
        logger.info("Pass 2: Extracting sub-locations for each region...")
        detailed_graph = await self._extract_sub_locations_async(
            top_level_graph,
            combined_summary
        )

        # Pass 3: Relationships
        logger.info("Pass 3: Extracting additional relationships...")
        final_graph = await self._extract_relationships_async(
            detailed_graph,
            combined_summary
        )

        # Pass 4: Verification
        logger.info("Pass 4: Verifying and refining...")
        final_graph = await self._verify_and_refine_async(final_graph, combined_summary)

        logger.info(f"=== Multi-Pass Complete: {len(final_graph['nodes'])} nodes, {len(final_graph['edges'])} edges ===")
        return final_graph

    async def _extract_top_level_async(self, combined_summary: str) -> dict:
        """Pass 1: Extract only top-level locations."""
        prompt = PromptFactory.create_top_level_extraction_prompt(combined_summary)
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context="Top-level extraction"
        )
        return result if result else {"nodes": [], "edges": []}

    async def _extract_sub_locations_async(
        self,
        top_level_graph: dict,
        combined_summary: str
    ) -> dict:
        """Pass 2: Extract sub-locations for each parent."""
        parent_candidates = [
            node for node in top_level_graph.get("nodes", [])
            if node.get("type") in ["Island", "Region", "City", "Building", "Cave System"]
        ]

        if not parent_candidates:
            logger.info("No parent candidates found for sub-location extraction")
            return top_level_graph

        existing_nodes_text = "\n".join([
            f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})"
            for n in top_level_graph.get("nodes", [])
        ])

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

        # Merge sub-graphs
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
        self,
        parent_id: str,
        parent_label: str,
        combined_summary: str,
        existing_nodes: str
    ) -> dict:
        """Extract sub-locations for a single parent."""
        prompt = PromptFactory.create_sub_location_extraction_prompt(
            parent_id, parent_label, combined_summary, existing_nodes
        )
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context=f"Sub-location extraction for {parent_label}"
        )
        return result if result else {"nodes": [], "edges": []}

    async def _extract_relationships_async(
        self,
        graph: dict,
        combined_summary: str
    ) -> dict:
        """Pass 3: Extract additional relationships."""
        nodes_text = "\n".join([
            f"- [{n['id']}] {n.get('label', 'Unknown')} ({n.get('type', 'Location')})"
            for n in graph.get("nodes", [])
        ])

        prompt = PromptFactory.create_relationship_extraction_prompt(nodes_text, combined_summary)
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context="Relationship extraction"
        )

        if not result:
            return graph

        # Merge new edges with deduplication
        existing_edges = graph.get("edges", [])
        new_edges = result.get("edges", [])

        seen_edges = {
            f"{e['source']}|{e.get('relation', 'related_to')}|{e['target']}"
            for e in existing_edges
        }

        for edge in new_edges:
            edge_key = f"{edge['source']}|{edge.get('relation', 'related_to')}|{edge['target']}"
            if edge_key not in seen_edges:
                existing_edges.append(edge)
                seen_edges.add(edge_key)

        graph["edges"] = existing_edges
        logger.info(f"  Relationship extraction: added {len(new_edges)} edges")
        return graph

    async def _verify_and_refine_async(self, graph: dict, combined_summary: str) -> dict:
        """Pass 4: Verification and refinement."""
        graph_text = format_graph_summary(graph)
        prompt = PromptFactory.create_multi_pass_verification_prompt(graph_text, combined_summary)
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context="Verification"
        )

        if not result:
            return graph

        if result.get("revised_edges"):
            for edge in result.get("revised_edges", []):
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


class SectionLocationMapper(BaseLLMProcessor):
    """Map shadow tree sections to location graph nodes."""

    def process(self, shadow_tree: list[dict], location_graph: dict) -> dict[str, list[str]]:
        """
        Synchronous entry point.

        Returns:
            Dict mapping section_id -> list of location_ids
        """
        async def _run_and_cleanup():
            try:
                return await self._process_async(shadow_tree, location_graph)
            finally:
                await self.close()

        return asyncio.run(_run_and_cleanup())

    async def _process_async(
        self,
        shadow_tree: list[dict],
        location_graph: dict
    ) -> dict[str, list[str]]:
        from utils.tree_traversal import collect_sections_with_context

        location_ids = [n["id"] for n in location_graph.get("nodes", [])]
        if not location_ids:
            logger.warning("No locations found in graph, skipping mapping.")
            return {}

        sections = collect_sections_with_context(shadow_tree)
        logger.info(f"Mapping {len(sections)} sections to {len(location_ids)} locations...")

        tasks = [self._map_section_to_locations(s, location_ids) for s in sections]
        results = await asyncio.gather(*tasks)

        # Organize results
        mapping = {}
        for sec_id, loc_ids in results:
            if loc_ids is None:
                logger.warning(f"Skipping section {sec_id} due to mapping error")
                continue
            valid_ids = [lid for lid in loc_ids if lid in location_ids]
            if valid_ids:
                mapping[sec_id] = valid_ids

        logger.info(f"Successfully mapped {len(mapping)} sections.")
        return mapping

    async def _map_section_to_locations(
        self,
        section_ctx: dict,
        location_ids: list[str]
    ) -> tuple:
        """Map a single section to locations with smart truncation."""
        child_titles_str = ", ".join(section_ctx.get('child_titles', [])) or "None"

        content = section_ctx.get('content', '')
        if len(content) > 3000:
            content = content[:3000] + "... [truncated]"

        context = (
            f"ID: {section_ctx['id']}\n"
            f"Parent Title: {section_ctx['parent_title']}\n"
            f"Title: {section_ctx['title']}\n"
            f"Content: {content}\n"
            f"Child Titles: {child_titles_str}"
        )

        # Truncate location list to prevent token overflow
        if len(location_ids) > 200:
            logger.warning(
                f"Section {section_ctx.get('id')}: Location list truncated from {len(location_ids)} to 200"
            )
            location_ids = location_ids[:200]

        loc_list_str = ", ".join(location_ids)
        prompt = PromptFactory.create_section_mapping_prompt(context, loc_list_str)

        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context=f"Section mapping for {section_ctx['id']}"
        )

        if result is None:
            return section_ctx["id"], None

        raw_result = result.get("location_ids") or result.get("location_id")

        if isinstance(raw_result, list):
            loc_ids = raw_result
        elif isinstance(raw_result, str):
            loc_ids = [raw_result]
        else:
            loc_ids = []

        return section_ctx["id"], loc_ids
