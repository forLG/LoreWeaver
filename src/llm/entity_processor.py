import asyncio
import json

from llm.base_processor import BaseLLMProcessor
from llm.prompt_factory import PromptFactory
from utils.graph_utils import deduplicate_graph
from utils.logger import logger
from utils.tree_traversal import collect_sections


class EntityProcessor(BaseLLMProcessor):
    """
    Extract entities and relationships from adventure text.

    Inherits async LLM functionality from BaseLLMProcessor.
    """

    def process(self, shadow_tree: list[dict], section_map: dict[str, list[str]]) -> dict:
        """
        Synchronous entry point for entity extraction.

        Args:
            shadow_tree: Parsed adventure tree structure
            section_map: Mapping of section IDs to location IDs

        Returns:
            Graph dict with 'nodes' and 'edges'
        """
        async def _run_and_cleanup():
            try:
                return await self._process_async(shadow_tree, section_map)
            finally:
                await self.close()

        return asyncio.run(_run_and_cleanup())

    async def _process_async(
        self,
        shadow_tree: list[dict],
        section_map: dict[str, list[str]]
    ) -> dict:
        # Collect all sections using shared utility
        sections = collect_sections(shadow_tree)
        logger.info(f"Starting Entity Extraction for {len(sections)} sections...")

        # Create tasks for sections with content and either links or locations
        tasks = [
            self._process_single_section(section, section_map.get(section["id"], []))
            for section in sections
            if section.get("content") and (section.get("links") or section_map.get(section["id"]))
        ]

        results = await asyncio.gather(*tasks)

        # Merge and deduplicate results using shared utility
        full_graph = {
            "nodes": [],
            "edges": []
        }
        for res in results:
            full_graph["nodes"].extend(res["nodes"])
            full_graph["edges"].extend(res["edges"])

        return deduplicate_graph(full_graph)

    async def _process_single_section(
        self,
        section: dict,
        location_ids: list[str]
    ) -> dict:
        """Extract entities from a single section."""
        # Prepare candidate entities from links
        candidates = self._prepare_candidates(section)

        # Skip if no meaningful context
        if not candidates and not location_ids:
            return {"nodes": [], "edges": []}

        # Build prompt and call LLM
        prompt = self._build_extraction_prompt(section, candidates, location_ids)

        # Use base class method for JSON response
        result = await self._call_llm_json_async(
            prompt,
            temperature=0.1,
            error_context=f"Entity extraction for section {section['id']}"
        )

        return result if result else {"nodes": [], "edges": []}

    def _prepare_candidates(self, section: dict, max_candidates: int = 50) -> list[dict]:
        """
        Prepare candidate entity list from section links.

        Args:
            section: The section to extract candidates from
            max_candidates: Maximum number of candidates to prevent token overflow
        """
        candidates = [
            {
                "tag": "party",
                "text": "The Characters / Party",
                "suggested_id": "party:characters"
            }
        ]

        for link in section.get("links", []):
            if link.get("tag") in ["creature", "item", "spell"]:
                candidates.append({
                    "tag": link["tag"],
                    "text": link["text"],
                    "suggested_id": f"{link['tag']}:{link['text'].lower().replace(' ', '_')}"
                })

        # Truncate if too many candidates to prevent token overflow
        if len(candidates) > max_candidates:
            logger.warning(
                f"Section {section.get('id')} has {len(candidates)} candidates, "
                f"truncating to {max_candidates}"
            )
            candidates = candidates[:max_candidates]

        return candidates

    def _build_extraction_prompt(
        self,
        section: dict,
        candidates: list[dict],
        location_ids: list[str]
    ) -> str:
        """Build the entity extraction prompt with smart truncation."""
        # Truncate content
        content = section.get("content", "")
        if len(content) > 4000:
            content = content[:4000] + "... [truncated]"

        # Truncate candidate list JSON to prevent token overflow
        candidate_list_str = json.dumps(candidates, indent=2, ensure_ascii=False)
        if len(candidate_list_str) > 10000:  # ~10K chars for candidates
            # Take first N candidates that fit within limit
            truncated_candidates = []
            current_length = 0
            for candidate in candidates:
                candidate_json = json.dumps(candidate, ensure_ascii=False)
                if current_length + len(candidate_json) > 10000:
                    break
                truncated_candidates.append(candidate)
                current_length += len(candidate_json) + 2  # +2 for comma/newline

            candidate_list_str = json.dumps(truncated_candidates, indent=2, ensure_ascii=False)
            logger.warning(
                f"Section {section.get('id')}: Candidate list truncated from {len(candidates)} "
                f"to {len(truncated_candidates)} due to size"
            )

        # Truncate location list to prevent token overflow
        if len(location_ids) > 100:
            logger.warning(
                f"Section {section.get('id')}: Location list truncated from {len(location_ids)} to 100"
            )
            location_ids = location_ids[:100]

        loc_list_str = ", ".join(location_ids)

        return PromptFactory.create_entity_enrichment_prompt(
            content,
            candidate_list_str,
            loc_list_str
        )
