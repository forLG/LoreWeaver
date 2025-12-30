"""
Small Model Processor - Entity-first pipeline for models with limited context (e.g., qwen3-8b).

Supports two modes:
1. Two-phase mode (default): NER → Relations
2. Single-phase unified mode (--single-phase): Entities + Events (heterogeneous graph)

Unified Heterogeneous Graph:
- Entity nodes: Static knowledge (locations, creatures, items)
- Event nodes: Dynamic narrative (encounters, discoveries, combat)
- Edges: Event → Entity (has_participant, occurs_at)

Pipeline (single-phase):
1. Unified entity + event extraction (no parent context)
2. Entity deduplication and resolution
3. Event processing with edge generation
4. Build heterogeneous graph

Pipeline (two-phase):
1. Independent NER extraction (no parent context)
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
    parse_unified_extraction,
)
from llm.prompt_factory import PromptFactory
from models import create_validated_graph
from utils.graph_utils import deduplicate_graph
from utils.logger import logger


class SmallModelProcessor(BaseLLMProcessor):
    """
    Entity-first processor optimized for small models (qwen3-8b, etc.).

    Modes:
    - Two-phase (default): Extract entities, then relations
    - Single-phase unified: Extract entities + events (heterogeneous graph)

    Key differences from standard processor:
    - Pure bottom-up: No parent context passed between nodes
    - Heterogeneous graph: Entity nodes + Event nodes
    - Events as peripheral nodes: Don't clutter core entity relationships
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
        max_tokens: int = 0,
        repetition_penalty: float | None = None
    ):
        """
        Initialize the small model processor.

        Uses unified entity+event extraction by default (heterogeneous graph).

        Args:
            api_key: OpenAI API key
            base_url: Custom API base URL (optional)
            model: Model name to use
            max_concurrent: Maximum concurrent requests
            output_dir: Directory to save intermediate debug outputs (optional)
            use_natural_language: Use natural language output instead of JSON (default: True)
            max_tokens: Maximum tokens per response (0 = no limit, recommended 2048 for small models)
            repetition_penalty: Repetition penalty for vLLM (1.0 = no penalty, 1.1-1.5 recommended)
        """
        super().__init__(api_key, base_url, model, max_concurrent, repetition_penalty)
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_natural_language = use_natural_language
        self.max_tokens = max_tokens if max_tokens > 0 else None

        if use_natural_language:
            logger.info("Natural language mode enabled (more robust for small models)")

        logger.info("Unified entity+event extraction mode enabled (heterogeneous graph)")

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

        Uses unified entity+event extraction (heterogeneous graph):
        - Phase 1: Unified entity + event extraction
        - Phase 2: Entity resolution
        - Phase 3: Event processing and edge generation
        - Phase 4: Build validated heterogeneous graph
        """
        logger.info("=" * 60)
        logger.info("SMALL MODEL PIPELINE: Unified Entity+Event Extraction")
        logger.info("=" * 60)

        return await self._process_single_phase(shadow_tree)

    async def _process_single_phase(
        self,
        shadow_tree: list[dict]
    ) -> dict:
        """
        Single-phase unified pipeline: Extract entities + events in one pass.
        Builds a heterogeneous graph with both entity and event nodes.
        """
        # ====================================================================
        # Phase 1: Unified entity + event extraction
        # ====================================================================
        logger.info("Phase 1: Unified entity + event extraction...")

        extraction_result = await self._unified_extraction(shadow_tree)

        all_entities = extraction_result["entities"]
        all_events = extraction_result["events"]

        logger.info(f"  Extracted {len(all_entities)} raw entities")
        logger.info(f"  Extracted {len(all_events)} raw events")

        # Save raw extractions for debugging
        self._save_debug("phase1_unified_entities", all_entities)
        self._save_debug("phase1_unified_events", all_events)

        # ====================================================================
        # Phase 2: Entity resolution
        # ====================================================================
        logger.info("Phase 2: Resolving entities...")

        # Deduplicate entities by ID
        unique_entities = self._deduplicate_entities(all_entities)
        logger.info(f"  Resolved to {len(unique_entities)} unique entities")

        # Build ID mapping for updating event references
        id_mapping = self._build_entity_id_mapping(all_entities, unique_entities)

        # ====================================================================
        # Phase 3: Event processing and edge generation
        # ====================================================================
        logger.info("Phase 3: Processing events and generating edges...")

        edges = []
        valid_events = []

        for event in all_events:
            # Update event entity references to use canonical IDs
            updated_event = self._update_event_entity_refs(event, id_mapping)

            # Generate edges from event to participants and location
            event_edges = self._generate_event_edges(updated_event)

            # Only add event if it has valid edges (connected to real entities)
            if event_edges:
                edges.extend(event_edges)
                valid_events.append(updated_event)

        logger.info(f"  Valid events after ID update: {len(valid_events)}")
        logger.info(f"  Generated {len(edges)} edges from events")

        # ====================================================================
        # Phase 4: Build and validate unified heterogeneous graph
        # ====================================================================
        logger.info("Phase 4: Building and validating unified heterogeneous graph...")

        try:
            # Create validated graph using Pydantic
            validated_graph = create_validated_graph(
                entities=unique_entities,
                events=valid_events,
                edges=edges
            )

            logger.info(f"  Validation passed: {len(validated_graph.nodes)} nodes ({len(validated_graph.get_entity_nodes())} entities + {len(validated_graph.get_event_nodes())} events)")
            logger.info(f"                   {len(validated_graph.edges)} edges")

            # Return as dict for JSON serialization
            return validated_graph.model_dump_dict()

        except Exception as e:
            logger.warning(f"  Pydantic validation failed: {e}")
            logger.warning("  Returning unvalidated graph (fallback)")

            # Fallback: return plain dict without validation
            all_nodes = []
            for entity in unique_entities:
                entity["node_type"] = "entity"
                all_nodes.append(entity)
            for event in valid_events:
                event["node_type"] = "event"
                all_nodes.append(event)

            return {
                "nodes": all_nodes,
                "edges": edges
            }

    async def _process_two_phase(
        self,
        shadow_tree: list[dict]
    ) -> dict:
        """
        Two-phase pipeline: Extract entities first, then relations.
        """
        # ====================================================================
        # Phase 1: Independent NER extraction (no parent context)
        # ====================================================================
        logger.info("Phase 1: Independent NER extraction (pure bottom-up)...")

        entity_layers = await self._independent_ner(shadow_tree)

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
        # Phase 3: Relation extraction (pure bottom-up, no parent context)
        # ====================================================================
        logger.info("Phase 3: Relation extraction (pure bottom-up)...")

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
    # Phase 1: Independent NER extraction (pure bottom-up)
    # ========================================================================

    async def _independent_ner(
        self,
        shadow_tree: list[dict]
    ) -> dict[str, list[dict]]:
        """
        Bottom-up traversal: Each node extracts entities independently.

        For each node:
        1. Extract entities from this node's content only (no parent context)
        2. Store entities for later aggregation

        Returns:
            Dict mapping node_id -> list of extracted entities
        """
        entity_layers = {}

        for root in shadow_tree:
            await self._ner_recursive(root, entity_layers)

        return entity_layers

    async def _ner_recursive(
        self,
        node: dict,
        entity_layers: dict[str, list[dict]]
    ) -> None:
        """
        Recursively extract entities independently (no parent context).
        """
        node_id = node.get("id", "unknown")

        # Extract entities for this node (no parent context)
        entities = await self._extract_entities_ner(node)

        entity_layers[node_id] = entities

        logger.debug(f"  NER for {node_id}: {len(entities)} entities")

        # Recurse to children (each extracts independently)
        if node.get("children"):
            for child in node["children"]:
                await self._ner_recursive(child, entity_layers)

    async def _extract_entities_ner(
        self,
        node: dict
    ) -> list[dict]:
        """
        Extract entities from a single node using NER.

        Each node extracts independently - no parent context is passed.
        Duplicates will be resolved during bottom-up aggregation.
        """
        title = node.get("title", "Untitled")
        node_id = node.get("id", "unknown")
        content = node.get("content", "")

        # Skip if content is empty or just whitespace
        if not content or not content.strip():
            logger.debug(f"Skipping NER for node '{title}' (id: {node_id}): empty content")
            return []

        # Check if content is too short
        if len(content.strip()) < 50:
            logger.debug(f"Skipping NER for node '{title}' (id: {node_id}): content too short ({len(content)} chars)")
            return []

        # Truncate content if too long
        original_len = len(content)
        if original_len > 2000:
            content = content[:2000] + "... [truncated]"

        # Choose prompt based on mode
        if self.use_natural_language:
            prompt = PromptFactory.create_ner_prompt_natural(
                title=title,
                content=content,
                known_entities=None  # No parent context in pure bottom-up
            )

            # Use regular LLM call (not JSON)
            # Higher temperature (0.9) breaks repetition loop in Qwen3
            # max_tokens prevents infinite output
            # stop sequences prevent empty template repetition
            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_tokens,
                enable_thinking=False,
                stop=[]  # No stop sequences for XML-tagged output
            )

            # Parse natural language output
            result = parse_ner_entities(raw_response)

            # Check if LLM returned a summary (no entities found)
            if result.get("summary"):
                summary = result["summary"]
                logger.warning(
                    f"No entities found in node '{title}' (id: {node_id}). "
                    f"Reason: {summary} "
                    f"[Content length: {original_len} chars]"
                )
                return []
        else:
            prompt = PromptFactory.create_ner_prompt(
                title=title,
                content=content,
                parent_context=None,  # No parent context in pure bottom-up
                known_entities=None
            )

            result = await self._call_llm_json_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                error_context=f"NER for node {node_id}",
                max_tokens=self.max_tokens,
                enable_thinking=False
            )

        entities = result.get("entities", []) if result else []

        # Warn if no entities extracted (but content exists)
        if not entities:
            logger.debug(f"No entities extracted from node '{title}' (id: {node_id})")

        return entities

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
            # stop sequences prevent empty template repetition
            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_tokens,
                enable_thinking=False,
                stop=["\n -> \n", "\n->\n"]  # Stop on empty mapping (compact format)
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
    # Phase 3: Relation extraction (pure bottom-up, no parent context)
    # ========================================================================

    async def _extract_relations_bidirectional(
        self,
        shadow_tree: list[dict],
        resolved_entities: dict[str, Any],
        alias_map: dict[str, str]
    ) -> dict[str, Any]:
        """
        Extract relations using bottom-up tree traversal.

        Each node extracts relations independently based on its content.
        Relations are discovered from the text, not from parent context.

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
                alias_map
            )
            all_edges.extend(edges)

        # Deduplicate edges
        unique_edges = self._deduplicate_edges(all_edges)

        return {"edges": unique_edges}

    async def _extract_relations_recursive(
        self,
        node: dict,
        entity_map: dict[str, dict],
        alias_map: dict[str, str]
    ) -> list[dict]:
        """
        Recursively extract relations independently (no parent context).
        """
        content = node.get("content", "")

        # Get entities mentioned in this node's content (using alias map)
        mentioned_entities = self._find_mentioned_entities(content, alias_map)

        # Extract relations for this node
        edges = await self._extract_relations_for_node(
            node,
            mentioned_entities,
            entity_map
        )

        # Recurse to children (each extracts independently)
        if node.get("children"):
            for child in node["children"]:
                child_edges = await self._extract_relations_recursive(
                    child,
                    entity_map,
                    alias_map
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
        entity_map: dict[str, dict]
    ) -> list[dict]:
        """
        Extract relations for a single node.

        Relations are extracted based only on entities mentioned in this node's content.
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

        # Build context: only entities mentioned in this node's content
        entities_text = "\n".join([
            f"- [{eid}] {entity_map[eid].get('label', 'Unknown')} ({entity_map[eid].get('type', 'Entity')})"
            for eid in mentioned_entities
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
            # stop sequences prevent empty template repetition
            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.9,
                top_p=0.95,
                max_tokens=self.max_tokens,
                enable_thinking=False,
                stop=["\nRelation: \n", "\nRelation:\n"]  # Stop on empty relation (compact format)
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

    # ========================================================================
    # Single-Phase Combined Extraction (Experimental)
    # ========================================================================

    async def _unified_extraction(
        self,
        shadow_tree: list[dict]
    ) -> dict[str, Any]:
        """
        Extract entities and events in a single pass per node (unified heterogeneous graph).

        Returns:
            {"entities": [...], "events": [...]}
        """
        all_entities = []
        all_events = []

        for root in shadow_tree:
            await self._unified_extraction_recursive(root, all_entities, all_events)

        return {"entities": all_entities, "events": all_events}

    async def _unified_extraction_recursive(
        self,
        node: dict,
        all_entities: list[dict],
        all_events: list[dict]
    ) -> None:
        """
        Recursively extract entities and events from each node.
        """
        title = node.get("title", "Untitled")
        node_id = node.get("id", "unknown")
        content = node.get("content", "")

        # Skip if content is empty
        if not content or not content.strip():
            logger.debug(f"Skipping combined extraction for node '{title}' (id: {node_id}): empty content")
        else:
            original_len = len(content)

            # Check if content is too short
            if len(content.strip()) < 50:
                logger.debug(f"Skipping combined extraction for node '{title}' (id: {node_id}): content too short ({original_len} chars)")
            else:
                # Truncate content if too long
                if original_len > 2000:
                    content = content[:2000] + "... [truncated]"
                    logger.warning(f"Truncated content for node '{title}' (id: {node_id}) from {original_len} to {len(content)} chars")

                # Extract entities and events in one call (unified extraction)
                if self.use_natural_language:
                    prompt = PromptFactory.create_unified_extraction_prompt_natural(
                        title=title,
                        content=content
                    )

                    raw_response = await self._call_llm_async(
                        prompt,
                        temperature=0.75,
                        top_p=0.90,
                        max_tokens=self.max_tokens,
                        enable_thinking=False,
                        stop=[]  # No specific stop sequences for XML-tagged output
                    )

                    result = parse_unified_extraction(raw_response)

                    # Check if LLM returned a summary (no entities/events found)
                    if result.get("summary"):
                        summary = result["summary"]
                        logger.warning(
                            f"No entities/events found in node '{title}' (id: {node_id}). "
                            f"Reason: {summary} "
                            f"[Content length: {original_len} chars]"
                        )
                        return  # Skip adding entities/events
                else:
                    # JSON mode not implemented for unified extraction yet
                    logger.warning("Unified extraction only supports natural language mode")
                    result = {"entities": [], "events": []}

                # Add entities with source node info
                for entity in result.get("entities", []):
                    entity["source_node"] = title
                    all_entities.append(entity)

                # Add events with source node info
                for event in result.get("events", []):
                    event["source_node"] = title
                    all_events.append(event)

                logger.debug(f"  Unified extraction for {title}: "
                            f"{len(result.get('entities', []))} entities, "
                            f"{len(result.get('events', []))} events")

        # Recurse to children
        if node.get("children"):
            for child in node["children"]:
                await self._unified_extraction_recursive(child, all_entities, all_events)

    def _build_entity_id_mapping(
        self,
        raw_entities: list[dict],
        unique_entities: list[dict]
    ) -> dict[str, str]:
        """
        Build a mapping from raw entity IDs to canonical entity IDs.

        After deduplication, multiple raw entities may map to the same canonical entity.
        This mapping is used to update relation source/target IDs.
        """
        mapping = {}

        # Group raw entities by their canonical ID
        for raw_entity in raw_entities:
            raw_id = raw_entity.get("id", "")

            # Find the canonical entity this maps to
            for unique_entity in unique_entities:
                unique_id = unique_entity.get("id", "")

                # Check if they're the same entity (by ID or label similarity)
                if raw_id == unique_id or raw_entity.get("label") == unique_entity.get("label"):
                    mapping[raw_id] = unique_id
                    break

        # For any unmapped entities, map to themselves
        for raw_entity in raw_entities:
            raw_id = raw_entity.get("id", "")
            if raw_id and raw_id not in mapping:
                mapping[raw_id] = raw_id

        return mapping

    def _update_relation_ids(
        self,
        relations: list[dict],
        id_mapping: dict[str, str]
    ) -> list[dict]:
        """
        Update relation source/target IDs to use canonical entity IDs.

        Args:
            relations: List of relations with possibly non-canonical entity IDs
            id_mapping: Mapping from raw IDs to canonical IDs

        Returns:
            List of relations with updated IDs
        """
        updated = []

        for relation in relations:
            new_relation = relation.copy()

            # Update source ID
            source_id = relation.get("source", "")
            new_relation["source"] = id_mapping.get(source_id, source_id)

            # Update target ID
            target_id = relation.get("target", "")
            new_relation["target"] = id_mapping.get(target_id, target_id)

            updated.append(new_relation)

        return updated

    def _update_event_entity_refs(
        self,
        event: dict,
        id_mapping: dict[str, str]
    ) -> dict:
        """
        Update event entity references to use canonical entity IDs.

        Args:
            event: Event dict with participants list and location field
            id_mapping: Mapping from raw IDs to canonical IDs

        Returns:
            Updated event dict with canonical entity IDs
        """
        updated_event = event.copy()

        # Update participant IDs
        participants = event.get("participants", [])
        updated_participants = []
        for pid in participants:
            canonical_id = id_mapping.get(pid, pid)
            updated_participants.append(canonical_id)
        updated_event["participants"] = updated_participants

        # Update location ID
        location = event.get("location")
        if location:
            updated_event["location"] = id_mapping.get(location, location)

        return updated_event

    def _generate_event_edges(
        self,
        event: dict
    ) -> list[dict]:
        """
        Generate edges from an event to its participants and location.

        Args:
            event: Event dict with id, type, participants, location, description

        Returns:
            List of edge dicts: {source, target, relation, description}
        """
        edges = []
        event_id = event.get("id", "")
        event_type = event.get("type", "event")
        description = event.get("description", "")

        if not event_id:
            return edges

        # Edge: event --[has_participant]--> entity
        for participant_id in event.get("participants", []):
            if participant_id:  # Skip empty participant IDs
                edges.append({
                    "source": event_id,
                    "target": participant_id,
                    "relation": "has_participant",
                    "description": f"Participant in {event_type}"
                })

        # Edge: event --[occurs_at]--> location
        location_id = event.get("location")
        if location_id:
            edges.append({
                "source": event_id,
                "target": location_id,
                "relation": "occurs_at",
                "description": description or f"{event_type} at this location"
            })

        return edges

# ========================================================================
# Graph Filtering Utilities (for heterogeneous graphs)
# ========================================================================

def filter_graph_to_entities(graph: dict) -> dict:
    """
    Filter a heterogeneous graph to only entity nodes (remove events).

    Args:
        graph: Heterogeneous graph with nodes and edges

    Returns:
        Filtered graph with only entity nodes and entity-entity edges
    """
    # Get all event node IDs
    event_ids = {
        node["id"] for node in graph.get("nodes", [])
        if node.get("node_type") == "event"
    }

    # Filter to only entity nodes
    entity_nodes = [
        node for node in graph.get("nodes", [])
        if node.get("node_type") != "event"
    ]

    # Filter edges to remove those involving events
    entity_edges = [
        edge for edge in graph.get("edges", [])
        if edge.get("source") not in event_ids
        and edge.get("target") not in event_ids
    ]

    return {
        "nodes": entity_nodes,
        "edges": entity_edges
    }

