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
import re
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
        # Also filters out meaningless entities and merges similar ones (cliff_top vs clifftop)
        id_mapping, meaningful_entities = self._build_entity_id_mapping(all_entities, unique_entities)
        logger.info(f"  After filtering and merging: {len(meaningful_entities)} meaningful entities")

        # ====================================================================
        # Phase 2.5: Location hierarchy extraction
        # ====================================================================
        logger.info("Phase 2.5: Extracting location hierarchies...")

        location_entities = [e for e in meaningful_entities if e.get("type") == "Location"]
        hierarchy_edges = await self._extract_location_hierarchies(shadow_tree, location_entities)
        logger.info(f"  Extracted {len(hierarchy_edges)} location hierarchy edges")

        # Save location hierarchies for debugging
        self._save_debug("phase2_5_location_hierarchies", hierarchy_edges)

        # ====================================================================
        # Phase 2.6: Semantic relation extraction
        # ====================================================================
        logger.info("Phase 2.6: Extracting semantic relations...")

        semantic_edges = await self._extract_semantic_relations(shadow_tree, meaningful_entities, id_mapping)
        logger.info(f"  Extracted {len(semantic_edges)} semantic relation edges")

        # Save semantic relations for debugging
        self._save_debug("phase2_6_semantic_relations", semantic_edges)

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

        # Add hierarchy and semantic edges to the edge list
        edges.extend(hierarchy_edges)
        edges.extend(semantic_edges)

        # Deduplicate all edges
        edges = self._deduplicate_edges(edges)

        logger.info(f"  Total edges after adding hierarchies and semantic relations: {len(edges)}")

        # ====================================================================
        # Phase 4: Build and validate unified heterogeneous graph
        # ====================================================================
        logger.info("Phase 4: Building and validating unified heterogeneous graph...")

        try:
            # Create validated graph using Pydantic with filtered meaningful entities
            validated_graph = create_validated_graph(
                entities=meaningful_entities,
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
            for entity in meaningful_entities:
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
            await self._unified_extraction_recursive(
                root, all_entities, all_events,
                parent_context=""
            )

        return {"entities": all_entities, "events": all_events}

    async def _unified_extraction_recursive(
        self,
        node: dict,
        all_entities: list[dict],
        all_events: list[dict],
        parent_context: str = ""
    ) -> None:
        """
        Recursively extract entities and events from each node.
        Passes accumulated entities as context for disambiguation.
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
                if original_len > 3000:
                    content = content[:3000] + "... [truncated]"
                    logger.warning(f"Truncated content for node '{title}' (id: {node_id}) from {original_len} to {len(content)} chars")

                # Build known entities text from previously extracted entities
                known_entities_text = self._build_known_entities_text(all_entities)

                # Extract entities and events in one call (unified extraction)
                if self.use_natural_language:
                    prompt = PromptFactory.create_unified_extraction_prompt_natural(
                        title=title,
                        content=content,
                        parent_context=parent_context,
                        known_entities=known_entities_text
                    )

                    raw_response = await self._call_llm_async(
                        prompt,
                        temperature=0.75,
                        top_p=0.90,
                        max_tokens=self.max_tokens,
                        enable_thinking=False,
                        stop=["</events>\n", "</events>"]  # Stop after closing tag
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
                # Build parent context for child
                child_parent_context = f"{parent_context} > {title}" if parent_context else title
                await self._unified_extraction_recursive(
                    child, all_entities, all_events,
                    parent_context=child_parent_context
                )

    def _build_known_entities_text(self, entities: list[dict]) -> str:
        """
        Build a compact text summary of known entities for context.
        Shows previously extracted entities to help with disambiguation.
        """
        if not entities:
            return ""

        # Group by type for cleaner output
        by_type: dict[str, list[dict]] = {
            "Location": [],
            "Creature": [],
            "Item": [],
            "Group": []
        }

        for e in entities:
            etype = e.get("type", "Unknown")
            if etype in by_type:
                by_type[etype].append(e)

        lines = []
        for etype, elist in by_type.items():
            if elist:
                lines.append(f"{etype}:")
                for e in elist[:20]:  # Limit to 20 per type to avoid token overflow
                    label = e.get("label", "Unknown")
                    eid = e.get("id", "")
                    is_generic = e.get("is_generic", False)
                    generic_mark = " (group)" if is_generic else ""
                    lines.append(f"  - [{eid}] {label}{generic_mark}")

        return "\n".join(lines)

    def _build_alias_map(self, entities: list[dict]) -> dict[str, str]:
        """
        Build an alias map from entity list.

        Creates a mapping from all aliases (including labels and IDs)
        to the entity ID, for text matching purposes.

        Returns:
            {alias_or_label_or_id: entity_id}
        """
        alias_map = {}
        for e in entities:
            eid = e.get("id", "")
            if not eid:
                continue

            # Add ID as its own alias
            alias_map[eid] = eid

            # Add label
            label = e.get("label", "")
            if label:
                alias_map[label] = eid

            # Add aliases
            for alias in e.get("aliases", []):
                if alias:
                    alias_map[alias] = eid

        return alias_map

    def _normalize_entity_label(self, label: str) -> str:
        """
        Normalize entity label for similarity comparison.

        Handles:
        - Underscore vs space: cliff_top -> cliff top
        - Case insensitivity
        - Extra whitespace
        - Common contractions: observation -> observatory (optional)

        Returns:
            Normalized label string
        """
        if not label:
            return ""

        # Replace underscores with spaces
        normalized = label.replace('_', ' ')

        # Convert to lowercase
        normalized = normalized.lower()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Remove possessive markers
        normalized = normalized.replace("'", "").replace("`", "")

        return normalized

    def _are_labels_similar(self, label1: str, label2: str) -> bool:
        """
        Check if two entity labels refer to the same entity.

        Similarity criteria:
        - Exact match after normalization
        - One is a substring of the other (after normalization)
        - Edit distance threshold (for typos like clifftop vs cliff_top)

        Examples:
        - cliff_top_observation ≈ clifftop_observation
        - dragon_s_rest ≈ dragons_rest
        """
        norm1 = self._normalize_entity_label(label1)
        norm2 = self._normalize_entity_label(label2)

        if not norm1 or not norm2:
            return False

        # Exact match after normalization
        if norm1 == norm2:
            return True

        # Substring match (one contains the other)
        # This handles cases like "dragon" vs "dragon_rest"
        if norm1 in norm2 or norm2 in norm1:
            # Require significant overlap (at least 50% of the longer string)
            min_len = min(len(norm1), len(norm2))
            max_len = max(len(norm1), len(norm2))
            if min_len / max_len >= 0.5:
                return True

        # Simple edit distance for close matches
        # cliff_top (9) vs clifftop (8) - should match
        distance = self._levenshtein_distance(norm1, norm2)
        max_len = max(len(norm1), len(norm2))

        # Allow up to 20% difference (e.g., 2 chars in a 10-char string)
        return bool(distance <= 2 or (distance / max_len) <= 0.2)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _is_meaningless_entity(self, entity: dict) -> bool:
        """
        Detect meaningless/generic placeholder entities that should be removed.

        Patterns to filter:
        - Numbered locations: location_1, location_2, adventure_location_1, area_1, a_1
        - Generic category names: adventure_locations, unknown_entities
        - Template placeholders: entity_1, node_1, object_1

        Returns:
            True if entity should be filtered out
        """
        eid = entity.get("id", "")
        label = entity.get("label", "")

        # Pattern 1: Generic numbered locations (location_1, adventure_location_2, area_3)
        numbered_patterns = [
            r'^location_\d+$',
            r'^location\d+$',
            r'^area_[a-zA-Z]\d+$',  # area_A1, area_b2
            r'^area\d+$',
            r'^adventure_location_\d+$',
            r'^adventure_location\d+$',
            r'^loc_\d+$',
            r'^loc\d+$',
            r'^zone_\d+$',
            r'^zone\d+$',
            r'^region_\d+$',
            r'^region\d+$',
            r'^section_\d+$',
            r'^section\d+$',
        ]

        for pattern in numbered_patterns:
            if re.match(pattern, eid.lower()):
                return True

        # Pattern 2: Generic category entities (no specific identity)
        generic_categories = [
            'adventure_locations',
            'unknown_entities',
            'unknown_locations',
            'generic_entities',
            'miscellaneous_entities',
            'other_entities',
            'various_locations',
            'several_locations',
            'multiple_entities',
        ]

        if eid.lower() in generic_categories:
            return True

        # Pattern 3: Template placeholders
        placeholder_patterns = [
            r'^entity_\d+$',
            r'^node_\d+$',
            r'^object_\d+$',
            r'^item_\d+$',
            r'^thing_\d+$',
            r'^placeholder_\d+$',
            r'^temp_\d+$',
            r'^unknown_\d+$',
        ]

        for pattern in placeholder_patterns:
            if re.match(pattern, eid.lower()):
                return True

        # Pattern 4: Labels that look like generic descriptions
        # E.g., "a location", "an area", "unknown entity"
        if label:
            label_lower = label.lower().strip()
            generic_label_patterns = [
                r'^an? (area|location|place|spot|region|zone)$',
                r'^unknown (area|location|place|entity|object)$',
                r'^generic (area|location|place|entity|object)$',
                r'^unspecified (area|location|place|entity|object)$',
                r'^unnamed (area|location|place|entity|object)$',
            ]

            for pattern in generic_label_patterns:
                if re.match(pattern, label_lower):
                    return True

        return False

    def _build_entity_id_mapping(
        self,
        raw_entities: list[dict],
        unique_entities: list[dict]
    ) -> tuple[dict[str, str], list[dict]]:
        """
        Build a mapping from raw entity IDs to canonical entity IDs.

        Improvements:
        - Uses fuzzy label similarity matching (handles cliff_top vs clifftop)
        - Filters out meaningless/placeholder entities (adventure_location_1, etc.)

        After deduplication, multiple raw entities may map to the same canonical entity.
        This mapping is used to update relation source/target IDs.

        Returns:
            Tuple of (mapping dict {raw_id: canonical_id, ...}, filtered_entities list)
        """
        mapping = {}

        # First, filter out meaningless entities from unique_entities
        meaningful_entities = []
        filtered_count = 0

        for entity in unique_entities:
            if self._is_meaningless_entity(entity):
                filtered_count += 1
                logger.debug(f"  Filtering meaningless entity: {entity.get('id')} ({entity.get('label')})")
            else:
                meaningful_entities.append(entity)

        if filtered_count > 0:
            logger.info(f"  Filtered out {filtered_count} meaningless/placeholder entities")

        # Build similarity index for faster matching
        # Group entities by normalized label prefix
        entity_groups = {}

        for canonical_entity in meaningful_entities:
            canonical_label = canonical_entity.get("label", "")

            # Create normalized key for grouping
            norm_label = self._normalize_entity_label(canonical_label)
            prefix = norm_label[:5] if len(norm_label) >= 5 else norm_label  # First 5 chars

            if prefix not in entity_groups:
                entity_groups[prefix] = []
            entity_groups[prefix].append(canonical_entity)

        # Group raw entities by their canonical ID
        for raw_entity in raw_entities:
            raw_id = raw_entity.get("id", "")
            raw_label = raw_entity.get("label", "")

            if not raw_id:
                continue

            # Skip meaningless entities in mapping (they won't be in meaningful_entities)
            if self._is_meaningless_entity(raw_entity):
                mapping[raw_id] = None  # Mark for removal
                continue

            # Find the canonical entity this maps to
            mapped = False

            # Check exact ID match first
            if any(raw_id == e.get("id") for e in meaningful_entities):
                mapping[raw_id] = raw_id
                mapped = True

            # Check exact label match
            if not mapped:
                for canonical_entity in meaningful_entities:
                    if raw_label == canonical_entity.get("label"):
                        mapping[raw_id] = canonical_entity.get("id")
                        mapped = True
                        break

            # Check fuzzy label similarity
            if not mapped:
                # Only check entities with similar prefix (optimization)
                norm_label = self._normalize_entity_label(raw_label)
                prefix = norm_label[:5] if len(norm_label) >= 5 else norm_label

                candidates = entity_groups.get(prefix, [])
                # Also check neighboring prefixes (in case of slight variations)
                for p in [prefix, prefix[:-1], prefix + 'a']:
                    candidates.extend(entity_groups.get(p, []))

                for canonical_entity in candidates:
                    canonical_label = canonical_entity.get("label", "")
                    if self._are_labels_similar(raw_label, canonical_label):
                        mapping[raw_id] = canonical_entity.get("id")
                        mapped = True
                        logger.debug(f"  Merged '{raw_id}' -> '{canonical_entity.get('id')}' "
                                   f"(similar labels: {raw_label} ≈ {canonical_label})")
                        break

            # For any unmapped entities, map to themselves
            if not mapped and raw_id not in mapping:
                mapping[raw_id] = raw_id

        # Remove entities marked for filtering (mapped to None)
        mapping = {k: v for k, v in mapping.items() if v is not None}

        return mapping, meaningful_entities

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
    # Location Hierarchy Extraction
    # ========================================================================

    async def _extract_location_hierarchies(
        self,
        shadow_tree: list[dict],
        location_entities: list[dict]
    ) -> list[dict]:
        """
        Extract location hierarchy (part_of) relationships.

        For each node, extract which locations contain other locations.
        """
        if not location_entities:
            return []

        all_edges = []
        entity_map = {e["id"]: e for e in location_entities}

        for root in shadow_tree:
            edges = await self._extract_hierarchies_recursive(root, entity_map)
            all_edges.extend(edges)

        # Deduplicate edges
        return self._deduplicate_edges(all_edges)

    async def _extract_hierarchies_recursive(
        self,
        node: dict,
        entity_map: dict[str, dict]
    ) -> list[dict]:
        """Recursively extract location hierarchies."""
        content = node.get("content", "")

        # Skip empty content
        if not content or not content.strip():
            hierarchy_edges = []
        else:
            # Build alias map from entity map
            alias_map = self._build_alias_map(entity_map)

            # Find locations mentioned in this node
            mentioned_locations = self._find_mentioned_entities(content, alias_map)

            # Only extract hierarchies if we have 2+ locations
            if len(mentioned_locations) >= 2:
                hierarchy_edges = await self._extract_hierarchy_for_node(
                    node, mentioned_locations, entity_map
                )
            else:
                hierarchy_edges = []

        # Recurse to children
        if node.get("children"):
            for child in node["children"]:
                child_edges = await self._extract_hierarchies_recursive(child, entity_map)
                hierarchy_edges.extend(child_edges)

        return hierarchy_edges

    async def _extract_hierarchy_for_node(
        self,
        node: dict,
        mentioned_locations: set[str],
        entity_map: dict[str, dict]
    ) -> list[dict]:
        """Extract hierarchy relationships for a single node."""
        title = node.get("title", "")
        content = node.get("content", "")

        # Truncate content
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        # Build context: only locations mentioned in this node's content
        locations_text = "\n".join([
            f"- [{lid}] {entity_map[lid].get('label', 'Unknown')} ({entity_map[lid].get('location_type', 'Location')})"
            for lid in mentioned_locations
            if lid in entity_map and entity_map[lid].get("type") == "Location"
        ])

        if not locations_text.strip():
            return []

        # Use hierarchy extraction prompt
        prompt = PromptFactory.create_location_hierarchy_prompt_natural(
            title=title,
            content=content,
            locations_text=locations_text
        )

        raw_response = await self._call_llm_async(
            prompt,
            temperature=0.7,
            top_p=0.90,
            max_tokens=self.max_tokens,
            enable_thinking=False,
            stop=[]
        )

        # Parse natural language output
        from llm.natural_parsers import parse_relations
        result = parse_relations(raw_response)

        return result.get("relations", []) if result else []

    # ========================================================================
    # Semantic Relation Extraction
    # ========================================================================

    async def _extract_semantic_relations(
        self,
        shadow_tree: list[dict],
        entities: list[dict],
        id_mapping: dict[str, str]
    ) -> list[dict]:
        """
        Extract semantic relationships between entities.

        Extracts domain-specific relations like inhabits, guards, patrols,
        wields, hidden_at, unlocks, etc.
        """
        if not entities:
            return []

        all_edges = []
        entity_map = {e["id"]: e for e in entities}

        for root in shadow_tree:
            edges = await self._extract_semantic_relations_recursive(root, entity_map, id_mapping)
            all_edges.extend(edges)

        # Deduplicate edges
        unique_edges = self._deduplicate_edges(all_edges)

        # Update entity IDs using id_mapping
        updated_edges = self._update_relation_ids(unique_edges, id_mapping)

        return updated_edges

    async def _extract_semantic_relations_recursive(
        self,
        node: dict,
        entity_map: dict[str, dict],
        id_mapping: dict[str, str]
    ) -> list[dict]:
        """Recursively extract semantic relations."""
        content = node.get("content", "")

        # Skip empty content
        if not content or not content.strip():
            semantic_edges = []
        else:
            # Build alias map from entity map
            alias_map = self._build_alias_map(entity_map)

            # Find entities mentioned in this node
            mentioned_entities = self._find_mentioned_entities(content, alias_map)

            # Only extract relations if we have 2+ entities
            if len(mentioned_entities) >= 2:
                semantic_edges = await self._extract_semantic_relations_for_node(
                    node, mentioned_entities, entity_map
                )
            else:
                semantic_edges = []

        # Recurse to children
        if node.get("children"):
            for child in node["children"]:
                child_edges = await self._extract_semantic_relations_recursive(
                    child, entity_map, id_mapping
                )
                semantic_edges.extend(child_edges)

        return semantic_edges

    async def _extract_semantic_relations_for_node(
        self,
        node: dict,
        mentioned_entities: set[str],
        entity_map: dict[str, dict]
    ) -> list[dict]:
        """Extract semantic relations for a single node."""
        title = node.get("title", "")
        content = node.get("content", "")

        # Skip if content is empty or just whitespace
        if not content or not content.strip():
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

        # Use semantic relation extraction prompt
        prompt = PromptFactory.create_semantic_relation_prompt_natural(
            title=title,
            content=content,
            entities_text=entities_text
        )

        raw_response = await self._call_llm_async(
            prompt,
            temperature=0.7,
            top_p=0.90,
            max_tokens=self.max_tokens,
            enable_thinking=False,
            stop=["\nRelation: \n", "\nRelation:\n"]
        )

        # Parse natural language output
        from llm.natural_parsers import parse_relations
        result = parse_relations(raw_response)

        return result.get("relations", []) if result else []

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

