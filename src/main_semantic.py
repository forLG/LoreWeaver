"""
LoreWeaver Semantic Pipeline (Small Model Entry Point)

This pipeline is designed for small models (qwen, llama, mistral, etc.) and focuses on:
- Semantic entity extraction with location subtypes
- Generic creature handling (zombies, kobolds, etc.)
- Location hierarchy extraction (part_of relations)
- Domain-specific relation extraction (inhabits, guards, patrols, etc.)

Usage:
    # Run full pipeline
    python -m main_semantic --input data/adventure-dosi.json

    # Use specific model
    python -m main_semantic --model qwen3-8b

    # Adjust concurrency
    python -m main_semantic --max-concurrent 5

    # Rerun, ignore cache
    python -m main_semantic --force

Environment Variables (.env):
    OPENAI_API_KEY     - Your LLM API key (required)
    OPENAI_BASE_URL    - API base URL (default: https://api.openai.com/v1)
    LLM_MODEL          - Model name (default: qwen3-8b)
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

import config_neo4j as config
from llm.small_model_processor import SmallModelProcessor
from preprocessor.adventure_parser import AdventureParser
from utils.logger import logger, setup_logger

# Load environment variables from .env file
load_dotenv()


def get_default_concurrency():
    """Sensible default based on model or environment."""
    model = os.getenv('LLM_MODEL', 'qwen3-8b').lower()
    env_concurrency = os.getenv('LLM_MAX_CONCURRENT')
    if env_concurrency:
        try:
            return int(env_concurrency)
        except ValueError:
            pass
    # Conservative defaults for local/vLLM deployments
    if any(x in model for x in ['qwen', 'local']):
        return 5
    return 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoreWeaver Semantic Pipeline (Small Model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        default='data/adventure-dosi.json',
        help='Input adventure file (default: data/adventure-dosi.json)'
    )
    parser.add_argument(
        '--output-dir',
        default=str(config.OUTPUT_DIR),
        help=f'Output directory (default: {config.OUTPUT_DIR})'
    )

    # LLM Config
    parser.add_argument(
        '--api-key',
        default=os.getenv('OPENAI_API_KEY'),
        help='LLM API key (default: from OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--base-url',
        default=os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com/v1'),
        help='LLM base URL (default: from OPENAI_BASE_URL env var)'
    )
    parser.add_argument(
        '--model',
        default=os.getenv('LLM_MODEL', 'qwen3-8b'),
        help='LLM model (default: from LLM_MODEL env var)'
    )

    # Concurrency options
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=get_default_concurrency(),
        help='Maximum concurrent LLM requests (default: auto-detected)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=int(os.getenv('LLM_MAX_TOKENS', '8192')),
        help='Maximum tokens per LLM response (default: 8192)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=float(os.getenv('LLM_REPETITION_PENALTY', '0')) if os.getenv('LLM_REPETITION_PENALTY') else None,
        help='Repetition penalty for vLLM (default: from env var, 1.0 = no penalty)'
    )

    # Control flags
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rerun all stages, ignore cached files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would run without executing'
    )

    return parser.parse_args()


class SemanticPipeline:
    """Semantic extraction pipeline for small models."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        exp_name = self.output_dir.name
        setup_logger(exp_name=exp_name)

        # File paths
        self.input_file = Path(args.input)
        self.parsed_file = self.output_dir / "adventure_parsed.json"
        self.semantic_graph_file = self.output_dir / "semantic_graph.json"

        # Validate API key
        if not self.args.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY in .env file or use --api-key"
            )

        # Log configuration
        logger.info(f"Model: {self.args.model}")
        logger.info(f"Max concurrent requests: {self.args.max_concurrent}")
        if self.args.max_concurrent > 20:
            logger.warning("High concurrency may overload local vLLM servers!")

    def run(self):
        """Run the semantic pipeline."""
        if self.args.dry_run:
            logger.info("[Dry Run] Would parse adventure and extract semantic graph")
            return

        # Stage 1: Parse adventure
        self._parse_adventure()

        # Stage 2: Extract semantic graph
        self._extract_semantic_graph()

    def _parse_adventure(self):
        """Stage 1: Parse adventure JSON into internal index."""
        if self.parsed_file.exists() and not self.args.force:
            logger.info(f"Found cached parsed data: {self.parsed_file}")
            return

        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        logger.info(f"Loading adventure from {self.input_file}...")
        with open(self.input_file, encoding='utf-8') as f:
            raw_data = json.load(f)

        # AdventureParser expects the full raw_data dict (with 'data' key inside)
        logger.info("Parsing adventure data...")
        parser = AdventureParser()
        parsed_data = parser.parse(raw_data)

        logger.info(f"Saving parsed data to {self.parsed_file}...")
        with open(self.parsed_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Parsed {len(parsed_data['internal_index'])} nodes")
        logger.info(f"Found {len(parsed_data['external_refs'])} external references")

    def _extract_semantic_graph(self):
        """Stage 2: Extract semantic graph using SmallModelProcessor."""
        # Check if output exists
        if self.semantic_graph_file.exists() and not self.args.force:
            logger.info(f"Found cached semantic graph: {self.semantic_graph_file}")
            self._print_graph_stats()
            return

        # Load parsed data
        logger.info(f"Loading parsed data from {self.parsed_file}...")
        with open(self.parsed_file, encoding='utf-8') as f:
            parsed_data = json.load(f)

        # Convert internal_index to tree format for SmallModelProcessor
        shadow_tree = self._internal_index_to_tree(parsed_data['internal_index'])
        logger.info(f"Built tree with {len(shadow_tree)} root nodes")

        # Initialize processor
        logger.info("Initializing SmallModelProcessor...")
        processor = SmallModelProcessor(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent,
            output_dir=self.output_dir / "semantic_debug",
            use_natural_language=True,
            max_tokens=self.args.max_tokens,
            repetition_penalty=self.args.repetition_penalty
        )

        # Extract semantic graph
        logger.info("Extracting semantic graph (entities + events + relations)...")
        semantic_graph = processor.process(shadow_tree, skip_summary=False)

        # Save output
        logger.info(f"Saving semantic graph to {self.semantic_graph_file}...")
        with open(self.semantic_graph_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_graph, f, indent=2, ensure_ascii=False)

        # Print statistics
        self._print_graph_stats(semantic_graph)

    def _internal_index_to_tree(self, internal_index: dict) -> list:
        """
        Convert internal_index to shadow_tree format for SmallModelProcessor.

        The internal_index is a flat dict of nodes with parent_id references.
        We need to convert it to a hierarchical tree structure.

        Args:
            internal_index: {node_id: {id, type, name, parent_id, children, text_content, ...}}

        Returns:
            List of root nodes, each with nested children
        """
        # Build node map
        nodes = {}
        for node_id, node_data in internal_index.items():
            nodes[node_id] = {
                "id": node_id,
                "title": node_data.get("name", "Untitled"),
                "type": node_data.get("type", "section"),
                "content": self._join_text_content(node_data.get("text_content", [])),
                "children": []
            }

        # Build hierarchy by linking children to parents
        roots = []
        for node_id, node in nodes.items():
            parent_id = internal_index[node_id].get("parent_id")
            if parent_id and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)

        return roots

    def _join_text_content(self, text_content: list[str]) -> str:
        """Join text content list into a single string."""
        if not text_content:
            return ""
        return "\n\n".join(text_content)

    def _print_graph_stats(self, graph: dict | None = None):
        """Print statistics about the semantic graph."""
        if graph is None:
            # Load from file
            with open(self.semantic_graph_file, encoding='utf-8') as f:
                graph = json.load(f)

        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        logger.info("=" * 60)
        logger.info("SEMANTIC GRAPH STATISTICS")
        logger.info("=" * 60)

        # Node counts
        entity_nodes = [n for n in nodes if n.get('node_type') == 'entity']
        event_nodes = [n for n in nodes if n.get('node_type') == 'event']

        logger.info(f"Total nodes: {len(nodes)}")
        logger.info(f"  Entity nodes: {len(entity_nodes)}")
        logger.info(f"  Event nodes: {len(event_nodes)}")
        logger.info(f"Total edges: {len(edges)}")

        # Entity type distribution
        logger.info("\nEntity type distribution:")
        type_counts = Counter(n.get("type", "Unknown") for n in entity_nodes)
        for node_type, count in type_counts.most_common():
            logger.info(f"  {node_type}: {count}")

        # Location subtypes
        location_nodes = [n for n in entity_nodes if n.get("type") == "Location"]
        if location_nodes:
            location_types = Counter(n.get("location_type", "Unknown") for n in location_nodes)
            logger.info("\nLocation subtypes:")
            for loc_type, count in location_types.most_common():
                logger.info(f"  {loc_type}: {count}")

        # Generic creatures
        creature_nodes = [n for n in entity_nodes if n.get("type") == "Creature"]
        generic_count = sum(1 for n in creature_nodes if n.get("is_generic", False))
        named_count = len(creature_nodes) - generic_count
        logger.info("\nCreature statistics:")
        logger.info(f"  Named creatures: {named_count}")
        logger.info(f"  Generic groups: {generic_count}")

        # Creature types
        creature_types = Counter(n.get("creature_type", "Unknown") for n in creature_nodes)
        if creature_types:
            logger.info("\nCreature type distribution:")
            for cr_type, count in creature_types.most_common():
                logger.info(f"  {cr_type}: {count}")

        # Event types
        if event_nodes:
            event_types = Counter(n.get("type", "Unknown") for n in event_nodes)
            logger.info("\nEvent type distribution:")
            for evt_type, count in event_types.most_common():
                logger.info(f"  {evt_type}: {count}")

        # Edge relation types
        edge_types = Counter(e.get("relation", "unknown") for e in edges)
        logger.info("\nEdge relation types:")
        for rel_type, count in edge_types.most_common():
            logger.info(f"  {rel_type}: {count}")

        logger.info("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    pipeline = SemanticPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
