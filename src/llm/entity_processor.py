import json
import asyncio
from typing import List, Dict, Any, Set
from openai import AsyncOpenAI
from src.utils.logger import logger
from src.llm.prompt_factory import PromptFactory

class EntityProcessor:
    """Processor for extracting entities and relationships using an LLM."""

    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat", max_concurrent: int = 100):
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def process(self, shadow_tree: List[Dict], section_map: Dict[str, List[str]]) -> Dict:
        """Synchronous entry point for processing the shadow tree.

        Args:
            shadow_tree: The shadow tree structure containing sections.
            section_map: A mapping of section IDs to location IDs.

        Returns:
            A dictionary representing the extracted graph with nodes and edges.
        """
        return asyncio.run(self._process_async(shadow_tree, section_map))

    async def _process_async(self, shadow_tree: List[Dict], section_map: Dict[str, List[str]]) -> Dict:
        # 1. Collect all sections to be processed
        sections = self._collect_sections(shadow_tree)
        logger.info(f"Starting Entity Extraction for {len(sections)} sections...")

        tasks = []
        for section in sections:
            # Get location IDs corresponding to this section
            loc_ids = section_map.get(section["id"], [])
            
            # Process only if section has content AND (has links OR has corresponding locations).
            # Even without links, if content is rich, we might attempt extraction (depending on strategy),
            # but current conservative strategy requires links or known locations.
            if section.get("content") and (section.get("links") or loc_ids):
                tasks.append(self._process_single_section(section, loc_ids))

        results = await asyncio.gather(*tasks)

        # 2. Merge results and deduplicate
        full_graph = {"nodes": [], "edges": []}
        for res in results:
            full_graph["nodes"].extend(res["nodes"])
            full_graph["edges"].extend(res["edges"])

        return self._deduplicate_graph(full_graph)

    async def _process_single_section(self, section: Dict, location_ids: List[str]) -> Dict:
        # 1. Prepare candidate entities (extracted from links)
        candidates = []

        # Add a player node to handle interactions with players
        candidates.append({
            "tag": "player",
            "text": "Players",
            "suggested_id": "players"
        })

        for link in section.get("links", []):
            # Filter interested tags
            if link.get("tag") in ["creature", "item"]:
                candidates.append({
                    "tag": link["tag"],
                    "text": link["text"],
                    # Pre-generate a suggested ID for LLM reference
                    "suggested_id": f"{link['text'].lower().replace(' ', '_')}"
                })
        
        # If no candidates and no location context, extraction is likely low value, skip to save tokens.
        if not candidates and not location_ids:
            return {"nodes": [], "edges": []}

        candidate_list_str = json.dumps(candidates, indent=2, ensure_ascii=False)
        loc_list_str = ", ".join(location_ids)

        # 2. Build Prompt
        prompt = PromptFactory.create_entity_enrichment_prompt(
            section.get("content", ""),
            candidate_list_str,
            loc_list_str
        )

        # 3. Call LLM
        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                data = json.loads(response.choices[0].message.content)
                return data
        except Exception as e:
            logger.error(f"Entity extraction failed for section {section['id']}: {e}")
            return {"nodes": [], "edges": []}

    def _collect_sections(self, nodes: List[Dict]) -> List[Dict]:
        """Recursively collects all Section nodes from the tree."""
        collected = []
        for node in nodes:
            collected.append(node)
            if "children" in node:
                collected.extend(self._collect_sections(node["children"]))
        return collected

    def _deduplicate_graph(self, graph: Dict) -> Dict:
        """Simple graph deduplication logic."""
        unique_nodes = {}
        for n in graph["nodes"]:
            if n["id"] not in unique_nodes:
                unique_nodes[n["id"]] = n
            else:
                # If new node has more info (e.g. label), update it
                if len(n.get("label", "")) > len(unique_nodes[n["id"]].get("label", "")):
                    unique_nodes[n["id"]] = n

        unique_edges = []
        seen_edges = set()
        for e in graph["edges"]:
            # Normalize edge key
            key = f"{e['source']}|{e['relationship']}|{e['target']}"
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {"nodes": list(unique_nodes.values()), "edges": unique_edges}