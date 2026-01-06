"""
LoreWeaver Unified Pipeline for Small Models (Qwen3-8B)

Usage:
    python -m src.qwen3_8b.main --input data/adventure-dosi.json
    python -m src.qwen3_8b.main --model qwen3-8b --max-concurrent 5
    python -m src.qwen3_8b.main --force

Environment Variables (.env):
    OPENAI_API_KEY          - LLM API key (required)
    OPENAI_BASE_URL         - API base URL
    LLM_MODEL               - Model name (default: qwen3-8b)
    LLM_MAX_CONCURRENT      - Max concurrent requests (default: 5)
    LLM_MAX_TOKENS          - Max tokens per response (default: 8192)
    LLM_REPETITION_PENALTY  - Repetition penalty for vLLM
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.qwen3_8b.llm.small_model_processor import SmallModelProcessor
from src.qwen3_8b.preprocessor.adventure_parser import AdventureParser
from src.qwen3_8b.preprocessor.entity_resolver import EntityResolver
from src.utils.logger import logger, setup_logger

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "qwen3"


def get_default_concurrency():
    model = os.getenv('LLM_MODEL', 'qwen3-8b').lower()
    env_concurrency = os.getenv('LLM_MAX_CONCURRENT')
    if env_concurrency:
        try:
            return int(env_concurrency)
        except ValueError:
            pass
    if any(x in model for x in ['qwen', 'local']):
        return 5
    return 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoreWeaver Unified Pipeline (Small Model)",
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
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )

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


class UnifiedPipeline:
    """Unified semantic extraction pipeline for small models."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        exp_name = self.output_dir.name
        setup_logger(exp_name=exp_name)

        self.input_file = Path(args.input)
        self.parsed_file = self.output_dir / "adventure_parsed.json"
        self.resolved_entities_file = self.output_dir / "adventure_resolved.json"
        self.semantic_graph_file = self.output_dir / "semantic_graph.json"

        if not self.args.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY in .env file or use --api-key"
            )

        logger.info(f"Model: {self.args.model}")
        logger.info(f"Max concurrent requests: {self.args.max_concurrent}")
        if self.args.max_concurrent > 20:
            logger.warning("High concurrency may overload local vLLM servers!")

    def run(self):
        if self.args.dry_run:
            logger.info("[Dry Run] Would parse adventure and extract semantic graph")
            return

        self._parse_adventure()
        self._extract_semantic_graph()

    def _parse_adventure(self):
        if self.parsed_file.exists() and not self.args.force:
            logger.info(f"Found cached parsed data: {self.parsed_file}")
            return

        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        logger.info(f"Loading adventure from {self.input_file}...")
        with open(self.input_file, encoding='utf-8') as f:
            raw_data = json.load(f)

        logger.info("Parsing adventure data...")
        parser = AdventureParser()
        parsed_data = parser.parse(raw_data)

        logger.info(f"Saving parsed data to {self.parsed_file}...")
        with open(self.parsed_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Parsed {len(parsed_data['internal_index'])} nodes")
        logger.info(f"Found {len(parsed_data['external_refs'])} external references")

    def _extract_semantic_graph(self):
        if self.semantic_graph_file.exists() and not self.args.force:
            logger.info(f"Found cached semantic graph: {self.semantic_graph_file}")
            self._print_graph_stats()
            return

        logger.info(f"Loading parsed data from {self.parsed_file}...")
        with open(self.parsed_file, encoding='utf-8') as f:
            parsed_data = json.load(f)

        if self.resolved_entities_file.exists() and not self.args.force:
            logger.info(f"Found cached resolved entities: {self.resolved_entities_file}")
            with open(self.resolved_entities_file, encoding='utf-8') as f:
                resolved_data = json.load(f)
            resolved_entities = list(resolved_data.get("resolved_entities", {}).values())
        else:
            logger.info("Resolving entities from preprocessor tags...")
            resolver = EntityResolver(data_dir=str(self.input_file.parent))
            resolved_data = resolver.resolve(parsed_data)

            logger.info(f"Saving resolved entities to {self.resolved_entities_file}...")
            with open(self.resolved_entities_file, 'w', encoding='utf-8') as f:
                json.dump(resolved_data, f, indent=2, ensure_ascii=False)

            resolved_entities = list(resolved_data.get("resolved_entities", {}).values())
            logger.info(f"Resolved {len(resolved_entities)} entities from tags")

        content_tree = self._internal_index_to_tree(parsed_data['internal_index'])
        logger.info(f"Built tree with {len(content_tree)} root nodes")

        logger.info("Initializing SmallModelProcessor...")
        processor = SmallModelProcessor(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent,
            output_dir=self.output_dir / "semantic_debug",
            max_tokens=self.args.max_tokens,
            repetition_penalty=self.args.repetition_penalty
        )

        logger.info("Extracting semantic graph (entities + events + relations)...")
        semantic_graph = processor.process(content_tree, skip_summary=False)

        logger.info(f"Saving semantic graph to {self.semantic_graph_file}...")
        with open(self.semantic_graph_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_graph, f, indent=2, ensure_ascii=False)

        self._print_graph_stats(semantic_graph)

    def _internal_index_to_tree(self, internal_index: dict) -> list:
        nodes = {}
        for node_id, node_data in internal_index.items():
            nodes[node_id] = {
                "id": node_id,
                "title": node_data.get("name", "Untitled"),
                "type": node_data.get("type", "section"),
                "content": self._join_text_content(node_data.get("text_content", [])),
                "children": []
            }

        roots = []
        for node_id, node in nodes.items():
            parent_id = internal_index[node_id].get("parent_id")
            if parent_id and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)

        return roots

    def _join_text_content(self, text_content: list[str]) -> str:
        if not text_content:
            return ""
        return "\n\n".join(text_content)

    def _print_graph_stats(self, graph: dict | None = None):
        if graph is None:
            with open(self.semantic_graph_file, encoding='utf-8') as f:
                graph = json.load(f)

        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        logger.info("=" * 60)
        logger.info("SEMANTIC GRAPH STATISTICS")
        logger.info("=" * 60)

        entity_nodes = [n for n in nodes if n.get('node_type') == 'entity']
        event_nodes = [n for n in nodes if n.get('node_type') == 'event']

        logger.info(f"Total nodes: {len(nodes)}")
        logger.info(f"  Entity nodes: {len(entity_nodes)}")
        logger.info(f"  Event nodes: {len(event_nodes)}")
        logger.info(f"Total edges: {len(edges)}")

        logger.info("\nEntity type distribution:")
        type_counts = Counter(n.get("type", "Unknown") for n in entity_nodes)
        for node_type, count in type_counts.most_common():
            logger.info(f"  {node_type}: {count}")

        if event_nodes:
            event_types = Counter(n.get("type", "Unknown") for n in event_nodes)
            logger.info("\nEvent type distribution:")
            for evt_type, count in event_types.most_common():
                logger.info(f"  {evt_type}: {count}")

        edge_types = Counter(e.get("relation", "unknown") for e in edges)
        logger.info("\nEdge relation types:")
        for rel_type, count in edge_types.most_common():
            logger.info(f"  {rel_type}: {count}")

        logger.info("=" * 60)


def main():
    args = parse_args()
    pipeline = UnifiedPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
