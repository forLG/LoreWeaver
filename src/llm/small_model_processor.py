"""
Small Model Processor - Entity-first pipeline for models with limited context (e.g., qwen3-8b).

Uses bidirectional tree traversal:
1. Top-down: Pass parent entity context to children for better extraction
2. Bottom-up: Aggregate and resolve entities at each level

Pipeline:
1. Top-down NER with parent context
2. Bottom-up entity aggregation and resolution
3. Relation extraction with known entities
4. Build spatial graph from relations
"""
import asyncio
import json
from pathlib import Path
from typing import Any

from llm.base_processor import BaseLLMProcessor
from llm.natural_parsers import (
    parse_entity_resolution,
    parse_ner_entities,
    parse_relations,
)
from llm.prompt_factory import PromptFactory
from utils.graph_utils import deduplicate_graph
from utils.logger import logger


class SmallModelProcessor(BaseLLMProcessor):
    """
    Entity-first processor optimized for small models (qwen3-8b, etc.).

    Key differences from standard processor:
    - Extracts entities first (NER), then relations
    - Uses tree structure for context propagation
    - Processes layer-by-layer to bound context size
    - Saves intermediate outputs for debugging
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 100,
        output_dir: str | Path | None = None,
        use_natural_language: bool = True,
        max_tokens: int = 0
    ):
        """
        Initialize the small model processor.

        Args:
            api_key: OpenAI API key
            base_url: Custom API base URL (optional)
            model: Model name to use
            max_concurrent: Maximum concurrent requests
            output_dir: Directory to save intermediate debug outputs (optional)
            use_natural_language: Use natural language output instead of JSON (default: True)
            max_tokens: Maximum tokens per response (0 = no limit, recommended 2048 for small models)
        """
        super().__init__(api_key, base_url, model, max_concurrent)
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_natural_language = use_natural_language
        self.max_tokens = max_tokens if max_tokens > 0 else None

        if use_natural_language:
            logger.info("Natural language mode enabled (more robust for small models)")

    def _save_debug(self, name: str, data: Any) -> None:
        """
        Save intermediate debug output to JSON file.

        Args:
            name: Name of the debug file (without extension)
            data: Data to serialize (must be JSON-serializable)
        """
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        debug_file = self.output_dir / f"debug_{name}.json"

        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"  Saved debug output: {debug_file.name}")

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def process(self, shadow_tree: list[dict], skip_summary: bool = False) -> dict:
        """Synchronous entry point for small model pipeline."""
        async def _run_and_cleanup():
            try:
                return await self._process_async(shadow_tree, skip_summary)
            finally:
                await self.close()

        return asyncio.run(_run_and_cleanup())

    async def _process_async(
        self,
        shadow_tree: list[dict],
        skip_summary: bool  # noqa: ARG002 - kept for interface compatibility
    ) -> dict:
        """
        Entity-first pipeline for small models.

        Phase 1: Top-down NER with parent context
        Phase 2: Bottom-up aggregation and resolution
        Phase 3: Relation extraction with known entities
        Phase 4: Build spatial graph
        """
        logger.info("=" * 60)
        logger.info("SMALL MODEL PIPELINE: Entity-first approach")
        logger.info("=" * 60)

        # ====================================================================
        # Phase 1: Top-down NER with parent context propagation
        # ====================================================================
        logger.info("Phase 1: Top-down NER with parent context...")

        entity_layers = await self._top_down_ner(shadow_tree)

        total_entities = sum(len(entities) for entities in entity_layers.values())
        logger.info(f"  Extracted {total_entities} raw entities across {len(entity_layers)} nodes")

        # Save raw entity layers for debugging
        self._save_debug("phase1_ner_raw", entity_layers)

        # ====================================================================
        # Phase 2: Bottom-up aggregation and entity resolution
        # ====================================================================
        logger.info("Phase 2: Bottom-up aggregation and resolution...")

        resolved_entities = await self._bottom_up_resolve(shadow_tree, entity_layers)

        logger.info(f"  Resolved to {len(resolved_entities['nodes'])} unique entities")

        # Build alias map for entity linking
        alias_map = self._build_alias_map(resolved_entities["nodes"])
        logger.info(f"  Built alias map with {len(alias_map)} entries")

        # Save resolved entities and alias map for debugging
        self._save_debug("phase2_resolved_entities", resolved_entities)
        self._save_debug("phase2_alias_map", alias_map)

        # ====================================================================
        # Phase 3: Relation extraction with known entities
        # ====================================================================
        logger.info("Phase 3: Relation extraction with known entities...")

        relations = await self._extract_relations_bidirectional(
            shadow_tree,
            resolved_entities,
            alias_map
        )

        logger.info(f"  Extracted {len(relations['edges'])} relations")

        # Save extracted relations for debugging
        self._save_debug("phase3_relations", relations)

        # ====================================================================
        # Phase 4: Build final spatial graph
        # ====================================================================
        logger.info("Phase 4: Building final spatial graph...")

        final_graph = {
            "nodes": resolved_entities["nodes"],
            "edges": relations["edges"]
        }

        # Final deduplication
        final_kg = deduplicate_graph(final_graph)

        logger.info(f"  Final graph: {len(final_kg.nodes)} nodes, {len(final_kg.edges)} edges")

        return final_kg.to_dict()

    # ========================================================================
    # Phase 1: Top-down NER with parent context
    # ========================================================================

    async def _top_down_ner(
        self,
        shadow_tree: list[dict],
        parent_entities: list[dict] | None = None
    ) -> dict[str, list[dict]]:
        """
        Top-down traversal: Extract entities with parent context.

        For each node:
        1. Extract entities from this node's content
        2. Pass parent entities as context (child can reference parent)
        3. Recurse to children with this node's entities as their parent context

        Returns:
            Dict mapping node_id -> list of extracted entities
        """
        entity_layers = {}

        parent_entities = parent_entities or []

        for root in shadow_tree:
            await self._ner_recursive(root, parent_entities, entity_layers)

        return entity_layers

    async def _ner_recursive(
        self,
        node: dict,
        parent_entities: list[dict],
        entity_layers: dict[str, list[dict]]
    ) -> None:
        """
        Recursively extract entities with parent context.
        """
        node_id = node.get("id", "unknown")

        # Build parent context string
        parent_context = self._format_entity_context(parent_entities)

        # Extract entities for this node
        entities = await self._extract_entities_ner(node, parent_context)

        entity_layers[node_id] = entities

        logger.debug(f"  NER for {node_id}: {len(entities)} entities (parent context: {len(parent_entities)})")

        # Recurse to children, passing this node's entities as their parent context
        if node.get("children"):
            # Merge parent entities with this node's entities for children
            child_parent_context = parent_entities + entities
            for child in node["children"]:
                await self._ner_recursive(child, child_parent_context, entity_layers)

    async def _extract_entities_ner(
        self,
        node: dict,
        parent_context: str
    ) -> list[dict]:
        """
        Extract entities from a single node using NER.

        Parent context helps child nodes reference entities mentioned in parents.
        """
        title = node.get("title", "Untitled")
        content = node.get("content", "")

        # Skip if content is empty or just whitespace
        if not content or not content.strip():
            logger.debug(f"Skipping NER for node {title}: empty content")
            return []

        # Truncate content if too long
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        # Choose prompt based on mode
        if self.use_natural_language:
            prompt = PromptFactory.create_ner_prompt_natural(
                title=title,
                content=content,
                known_entities=parent_context
            )

            # Use regular LLM call (not JSON)
            # Higher temperature (0.9) breaks repetition loop in Qwen3
            # max_tokens prevents infinite output
            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_tokens,
                enable_thinking=False
            )

            # Parse natural language output
            result = parse_ner_entities(raw_response)
        else:
            prompt = PromptFactory.create_ner_prompt(
                title=title,
                content=content,
                parent_context=parent_context,
                known_entities=parent_context
            )

            result = await self._call_llm_json_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                error_context=f"NER for node {node.get('id')}",
                max_tokens=self.max_tokens,
                enable_thinking=False
            )

        return result.get("entities", []) if result else []

    def _format_entity_context(self, entities: list[dict]) -> str:
        """Format entity list as context string for LLM."""
        if not entities:
            return "No previously known entities."

        lines = ["Known entities from parent/previous sections:"]
        for e in entities:
            eid = e.get("id", "unknown")
            label = e.get("label", "Unknown")
            etype = e.get("type", "Entity")
            lines.append(f"  - [{eid}] {label} ({etype})")

        return "\n".join(lines)

    # ========================================================================
    # Phase 2: Bottom-up aggregation and entity resolution
    # ========================================================================

    async def _bottom_up_resolve(
        self,
        shadow_tree: list[dict],
        entity_layers: dict[str, list[dict]]
    ) -> dict[str, Any]:
        """
        Bottom-up traversal: Aggregate and resolve entities.

        For each node (from leaves to root):
        1. Collect entities from this node and all descendants
        2. Use LLM to resolve duplicates within this subtree
        3. Pass resolved entities up to parent

        Returns:
            Dict with 'nodes' (unique entities) and 'node_to_entities' mapping
        """
        all_resolved = []

        for root in shadow_tree:
            resolved = await self._resolve_recursive(root, entity_layers)
            all_resolved.extend(resolved)

        # Final deduplication across all roots
        unique_nodes = self._deduplicate_entities(all_resolved)

        return {"nodes": unique_nodes}

    async def _resolve_recursive(
        self,
        node: dict,
        entity_layers: dict[str, list[dict]]
    ) -> list[dict]:
        """
        Recursively resolve entities from leaves to root.
        """
        node_id = node.get("id", "unknown")
        my_entities = entity_layers.get(node_id, [])

        # First, resolve all children
        child_entity_lists = []
        if node.get("children"):
            for child in node["children"]:
                child_entities = await self._resolve_recursive(child, entity_layers)
                child_entity_lists.append(child_entities)

        # Flatten child entities
        all_child_entities = []
        for lst in child_entity_lists:
            all_child_entities.extend(lst)

        # Merge my entities with child entities
        all_entities = my_entities + all_child_entities

        # If we have many entities, use LLM to resolve duplicates
        if len(all_entities) > 10:
            resolved = await self._llm_resolve_entities(all_entities, node_id)
        else:
            # Simple ID-based deduplication for small sets
            resolved = self._deduplicate_entities(all_entities)

        return resolved

    async def _llm_resolve_entities(
        self,
        entities: list[dict],
        context_id: str
    ) -> list[dict]:
        """
        Use LLM to resolve duplicate entities.
        """
        entities_text = "\n".join([
            f"- [{e.get('id')}] {e.get('label')} ({e.get('type')})"
            for e in entities
        ])

        # Choose prompt based on mode
        if self.use_natural_language:
            prompt = PromptFactory.create_entity_resolution_prompt_natural(entities_text)

            # Use regular LLM call (not JSON)
            # Higher temperature (0.9) breaks repetition loop in Qwen3
            # max_tokens prevents infinite output
            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_tokens,
                enable_thinking=False
            )

            # Parse natural language output
            mapping = parse_entity_resolution(raw_response)
        else:
            prompt = PromptFactory.create_entity_resolution_prompt(entities_text)

            mapping = await self._call_llm_json_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                error_context=f"Entity resolution for {context_id}",
                max_tokens=2048,
                enable_thinking=False
            )

        if not mapping:
            return self._deduplicate_entities(entities)

        # Apply mapping
        return self._apply_entity_mapping(entities, mapping)

    def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """Simple ID-based deduplication (keep longest label)."""
        unique = {}
        for e in entities:
            eid = e.get("id")
            if not eid:
                continue
            if eid not in unique:
                unique[eid] = e
            else:
                # Keep the version with more complete label
                if len(e.get("label", "")) > len(unique[eid].get("label", "")):
                    unique[eid] = e

        return list(unique.values())

    def _apply_entity_mapping(
        self,
        entities: list[dict],
        mapping: dict[str, str]
    ) -> list[dict]:
        """Apply entity ID mapping."""
        seen = {}
        for e in entities:
            old_id = e.get("id")
            if not old_id:
                continue

            new_id = mapping.get(old_id, old_id)
            e["id"] = new_id

            if new_id not in seen:
                seen[new_id] = e
            else:
                # Keep longer label
                if len(e.get("label", "")) > len(seen[new_id].get("label", "")):
                    seen[new_id] = e

        return list(seen.values())

    def _build_alias_map(self, entities: list[dict]) -> dict[str, str]:
        """
        Build alias → canonical_id map for entity linking.

        For each entity, collect:
        - Primary label (normalized)
        - Aliases (normalized)
        - ID (normalized)

        Normalization: lowercase, underscores -> spaces
        This ensures consistent matching regardless of LLM output format.
        """
        alias_map = {}

        for entity in entities:
            eid = entity.get("id")
            label = entity.get("label", "")

            if not eid:
                continue

            def normalize(s: str) -> str:
                """Normalize string for matching: lowercase + underscores to spaces"""
                return s.lower().replace('_', ' ')

            # Add primary label (normalized)
            alias_map[normalize(label)] = eid

            # Add explicit aliases if present
            aliases = entity.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_map[normalize(alias)] = eid

            # Add ID as alias (normalized)
            alias_map[normalize(eid)] = eid

            # Add shortened versions (bigrams from label)
            words = normalize(label).split()
            if len(words) > 1:
                # Add bigrams for multi-word names
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    alias_map[bigram] = eid

        return alias_map

    # ========================================================================
    # Phase 3: Bidirectional relation extraction
    # ========================================================================

    async def _extract_relations_bidirectional(
        self,
        shadow_tree: list[dict],
        resolved_entities: dict[str, Any],
        alias_map: dict[str, str]
    ) -> dict[str, Any]:
        """
        Extract relations using bidirectional tree traversal.

        Top-down: Parent context helps identify hierarchical relations (part_of)
        Bottom-up: Child context helps identify composition relations

        Returns:
            Dict with 'edges' (list of relations)
        """
        all_edges = []

        # Build entity lookup
        entity_map = {e["id"]: e for e in resolved_entities["nodes"]}

        for root in shadow_tree:
            edges = await self._extract_relations_recursive(
                root,
                entity_map,
                alias_map,
                parent_entities=[]
            )
            all_edges.extend(edges)

        # Deduplicate edges
        unique_edges = self._deduplicate_edges(all_edges)

        return {"edges": unique_edges}

    async def _extract_relations_recursive(
        self,
        node: dict,
        entity_map: dict[str, dict],
        alias_map: dict[str, str],
        parent_entities: list[str]
    ) -> list[dict]:
        """
        Recursively extract relations with parent/child context.
        """
        content = node.get("content", "")

        # Get entities mentioned in this node's content (using alias map)
        mentioned_entities = self._find_mentioned_entities(content, alias_map)

        # Extract relations for this node
        edges = await self._extract_relations_for_node(
            node,
            mentioned_entities,
            entity_map,
            parent_entities
        )

        # Recurse to children
        if node.get("children"):
            # Pass this node's entities as parent context for children
            child_parent_entities = parent_entities + list(mentioned_entities)
            for child in node["children"]:
                child_edges = await self._extract_relations_recursive(
                    child,
                    entity_map,
                    alias_map,
                    child_parent_entities
                )
                edges.extend(child_edges)

        return edges

    def _find_mentioned_entities(
        self,
        text: str,
        alias_map: dict[str, str]
    ) -> set[str]:
        """
        Find which entities are mentioned in text using alias matching.

        Normalizes text for matching (underscores <-> spaces, case-insensitive).
        """
        mentioned = set()

        # Normalize text: replace underscores with spaces, lowercase
        # This handles: area_D1 <-> area D1, Dragon_s_rest <-> dragon s rest
        text_normalized = text.lower().replace('_', ' ')

        # Check all aliases for direct substring match
        for alias, entity_id in alias_map.items():
            if alias in text_normalized:
                mentioned.add(entity_id)

        return mentioned

    async def _extract_relations_for_node(
        self,
        node: dict,
        mentioned_entities: set[str],
        entity_map: dict[str, dict],
        parent_entities: list[str]
    ) -> list[dict]:
        """
        Extract relations for a single node.
        """
        if len(mentioned_entities) < 2:
            return []

        title = node.get("title", "")
        content = node.get("content", "")

        # Skip if content is empty or just whitespace
        if not content or not content.strip():
            logger.debug(f"Skipping relation extraction for node {title}: empty content")
            return []

        # Truncate content
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        # Build context: mentioned entities + parent entities
        entity_context = list(mentioned_entities) + parent_entities
        entity_context = list(set(entity_context))  # Deduplicate

        entities_text = "\n".join([
            f"- [{eid}] {entity_map[eid].get('label', 'Unknown')} ({entity_map[eid].get('type', 'Entity')})"
            for eid in entity_context
            if eid in entity_map
        ])

        # Choose prompt based on mode
        if self.use_natural_language:
            prompt = PromptFactory.create_relation_extraction_prompt_natural(
                title=title,
                content=content,
                entities_text=entities_text
            )

            # Use regular LLM call (not JSON)
            # Higher temperature (0.9) breaks repetition loop in Qwen3
            # max_tokens prevents infinite output
            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_tokens,
                enable_thinking=False
            )

            # Parse natural language output
            result = parse_relations(raw_response)
        else:
            prompt = PromptFactory.create_relation_extraction_prompt(
                title=title,
                content=content,
                entities_text=entities_text
            )

            result = await self._call_llm_json_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                error_context=f"Relation extraction for node {node.get('id')}",
                max_tokens=2048,
                enable_thinking=False
            )

        return result.get("relations", []) if result else []

    def _deduplicate_edges(self, edges: list[dict]) -> list[dict]:
        """Remove duplicate edges (source + relation + target)."""
        seen = set()
        unique = []

        for e in edges:
            key = f"{e.get('source')}|{e.get('relation')}|{e.get('target')}"
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique
