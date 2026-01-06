"""
Small Model Processor - Entity-first pipeline for models with limited context (e.g., qwen3-8b).

This processor uses a "content tree" built from AdventureParser's internal_index,
NOT the ShadowTreeBuilder used in the large model pipeline.

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

# Use project root logger
import sys
from pathlib import Path
from typing import Any

from src.qwen3_8b.llm.base_processor import BaseLLMProcessor
from src.qwen3_8b.llm.natural_parsers import (
    parse_unified_extraction,
)
from src.qwen3_8b.llm.prompt_factory import PromptFactory
from src.qwen3_8b.utils.graph_schemas import create_validated_graph

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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
            max_tokens: Maximum tokens per response (0 = no limit, recommended 2048 for small models)
            repetition_penalty: Repetition penalty for vLLM (1.0 = no penalty, 1.1-1.5 recommended)
        """
        super().__init__(api_key, base_url, model, max_concurrent, repetition_penalty)
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_tokens = max_tokens if max_tokens > 0 else None

        logger.info("Natural language mode enabled")

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

    def process(self, content_tree: list[dict], skip_summary: bool = False) -> dict:
        """
        Synchronous entry point for small model pipeline.

        Args:
            content_tree: Tree of nodes with content to extract from (built from internal_index)
            skip_summary: Whether to skip summarization (kept for compatibility)
        """
        async def _run_and_cleanup():
            try:
                return await self._process_async(content_tree, skip_summary)
            finally:
                await self.close()

        return asyncio.run(_run_and_cleanup())

    async def _process_async(
        self,
        content_tree: list[dict],
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

        return await self._process_single_phase(content_tree)

    async def _process_single_phase(
        self,
        content_tree: list[dict]
    ) -> dict:
        """
        Single-phase unified pipeline: Extract entities + events in one pass.
        Builds a heterogeneous graph with both entity and event nodes.

        Args:
            content_tree: Tree of nodes with content to extract from (built from internal_index)
        """
        # ====================================================================
        # Phase 1: Unified entity + event extraction
        # ====================================================================
        logger.info("Phase 1: Unified entity + event extraction...")

        extraction_result = await self._unified_extraction(content_tree)

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
        hierarchy_edges = await self._extract_location_hierarchies(content_tree, location_entities)
        logger.info(f"  Extracted {len(hierarchy_edges)} location hierarchy edges")

        # Save location hierarchies for debugging
        self._save_debug("phase2_5_location_hierarchies", hierarchy_edges)

        # ====================================================================
        # Phase 2.6: Semantic relation extraction
        # ====================================================================
        logger.info("Phase 2.6: Extracting semantic relations...")

        semantic_edges = await self._extract_semantic_relations(content_tree, meaningful_entities, id_mapping)
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

            # Filter edges to only include references to existing nodes
            valid_node_ids = {n.get("id") for n in all_nodes if n.get("id")}
            filtered_edges = []
            removed_count = 0

            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                if source in valid_node_ids and target in valid_node_ids:
                    filtered_edges.append(edge)
                else:
                    removed_count += 1

            if removed_count > 0:
                logger.warning(f"  Filtered out {removed_count} edges with broken references (from {len(edges)} total)")

            return {
                "nodes": all_nodes,
                "edges": filtered_edges
            }

    # ========================================================================
    # Single-Phase Combined Extraction (Experimental)
    # ========================================================================

    async def _unified_extraction(
        self,
        content_tree: list[dict]
    ) -> dict[str, Any]:
        """
        Extract entities and events in a single pass per node (unified heterogeneous graph).

        Args:
            content_tree: Tree of nodes with content to extract from (built from internal_index)

        Returns:
            {"entities": [...], "events": [...]}
        """
        all_entities = []
        all_events = []

        for root in content_tree:
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
        """
        title = node.get("title", "Untitled")
        node_id = node.get("id", "unknown")
        content = node.get("content", "")

        # Extract entities and events from this node
        result = {"entities": [], "events": []}

        # Skip if content is empty
        if not content or not content.strip():
            logger.debug(f"Skipping combined extraction for node '{title}' (id: {node_id}): empty content")
        elif len(content.strip()) < 50:
            # Content too short for extraction
            logger.debug(f"Skipping combined extraction for node '{title}' (id: {node_id}): content too short ({len(content)} chars)")
        else:
            # Content is long enough - proceed with extraction
            original_len = len(content)

            # Truncate content if too long
            if original_len > 3000:
                content = content[:3000] + "... [truncated]"
                logger.warning(f"Truncated content for node '{title}' (id: {node_id}) from {original_len} to {len(content)} chars")

            # Extract entities and events in one call (unified extraction)
            prompt = PromptFactory.create_unified_extraction_prompt_natural(
                title=title,
                content=content,
                parent_context=parent_context
            )

            raw_response = await self._call_llm_async(
                prompt,
                temperature=0.75,
                top_p=0.90,
                max_tokens=self.max_tokens,
                enable_thinking=False,
                stop=None  # Don't stop early - let LLM complete the response
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
                # Keep result as empty - children will still be processed

            # Add entities with source node info
            for entity in result.get("entities", []):
                entity["source_node"] = title
                entity["extraction_method"] = "llm"  # Mark as LLM-extracted
                all_entities.append(entity)

            # Add events with source node info
            for event in result.get("events", []):
                event["source_node"] = title
                all_events.append(event)

            logger.debug(f"  Unified extraction for {title}: "
                        f"{len(result.get('entities', []))} entities, "
                        f"{len(result.get('events', []))} events")

        # Recurse to children (ALWAYS happens, even if this node had no entities)
        if node.get("children"):
            for child in node["children"]:
                # Build parent context for child
                child_parent_context = f"{parent_context} > {title}" if parent_context else title
                await self._unified_extraction_recursive(
                    child, all_entities, all_events,
                    parent_context=child_parent_context
                )

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
            raw_aliases = {a.lower() for a in raw_entity.get("aliases", [])}

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

            # Check alias match (raw entity's aliases match canonical label)
            if not mapped and raw_aliases:
                for canonical_entity in meaningful_entities:
                    canonical_label = canonical_entity.get("label", "").lower()
                    if canonical_label in raw_aliases:
                        mapping[raw_id] = canonical_entity.get("id")
                        mapped = True
                        logger.debug(f"  Merged '{raw_id}' -> '{canonical_entity.get('id')}' "
                                   f"(alias match: '{canonical_label}' in {raw_aliases})")
                        break

            # Check canonical entity's aliases match raw label
            if not mapped and raw_label:
                for canonical_entity in meaningful_entities:
                    canonical_aliases = {a.lower() for a in canonical_entity.get("aliases", [])}
                    if raw_label.lower() in canonical_aliases:
                        mapping[raw_id] = canonical_entity.get("id")
                        mapped = True
                        logger.debug(f"  Merged '{raw_id}' -> '{canonical_entity.get('id')}' "
                                   f"(reverse alias match: '{raw_label}' in {canonical_aliases})")
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

    def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """
        Deduplicate entities by ID, keeping the one with more complete data.

        Args:
            entities: List of entity dicts

        Returns:
            Deduplicated entity list
        """
        unique_entities: dict[str, dict] = {}
        for entity in entities:
            entity_id = entity.get("id", "")
            if not entity_id:
                continue
            if entity_id not in unique_entities:
                unique_entities[entity_id] = entity
            else:
                existing = unique_entities[entity_id]
                # Keep the version with more complete label
                if len(entity.get("label", "")) > len(existing.get("label", "")):
                    unique_entities[entity_id] = entity
                # Merge aliases
                existing_aliases = set(existing.get("aliases", []))
                new_aliases = set(entity.get("aliases", []))
                if new_aliases - existing_aliases:
                    unique_entities[entity_id]["aliases"] = list(existing_aliases | new_aliases)
        return list(unique_entities.values())

    def _deduplicate_edges(self, edges: list[dict]) -> list[dict]:
        """
        Deduplicate edges by (source, relation, target) tuple.

        Args:
            edges: List of edge dicts

        Returns:
            Deduplicated edge list
        """
        unique_edges: list[dict] = []
        seen_edges: set[str] = set()
        for edge in edges:
            edge_key = f"{edge.get('source')}|{edge.get('relation')}|{edge.get('target')}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)
        return unique_edges

    # ========================================================================
    # Location Hierarchy Extraction
    # ========================================================================

    async def _extract_location_hierarchies(
        self,
        content_tree: list[dict],
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

        for root in content_tree:
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
            # Build alias map from entity map (convert dict values to list)
            alias_map = self._build_alias_map(list(entity_map.values()))

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
            f"- [{lid}] {entity_map[lid].get('label', 'Unknown')}"
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
        from src.qwen3_8b.llm.natural_parsers import parse_relations
        result = parse_relations(raw_response)

        return result.get("relations", []) if result else []

    # ========================================================================
    # Semantic Relation Extraction
    # ========================================================================

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

    async def _extract_semantic_relations(
        self,
        content_tree: list[dict],
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

        for root in content_tree:
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
            # Build alias map from entity map (convert dict values to list)
            alias_map = self._build_alias_map(list(entity_map.values()))

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
        prompt = PromptFactory.create_semantic_relations_prompt_natural(
            title=title,
            content=content,
            known_entities_text=entities_text
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
        from src.qwen3_8b.llm.natural_parsers import parse_relations
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

