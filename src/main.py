"""
LoreWeaver Main Pipeline

Usage:
    # Run all stages
    python -m main --stage all

    # Run specific stages
    python -m main --stage shadow
    python -m main --stage spatial
    python -m main --stage section-map
    python -m main --stage entity

    # Run multiple stages
    python -m main --stage spatial --stage entity

    # Rerun all stages, ignore cache
    python -m main --stage all --force

    # Preview what would run
    python -m main --stage all --dry-run

    # Multi-pass mode for smaller models (qwen3-8b, etc.)
    python -m main --stage all --multi-pass

    # Adjust concurrency for local vLLM deployments
    python -m main --stage all --max-concurrent 5

Environment Variables (.env):
    OPENAI_API_KEY     - Your LLM API key (required)
    OPENAI_BASE_URL    - API base URL (default: https://api.openai.com/v1)
    LLM_MODEL          - Model name (default: gpt-4o)
"""
import argparse
import os
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

from builder.shadow_builder import ShadowTreeBuilder
from llm.spatial_processor import SpatialTopologyProcessor, SectionLocationMapper
from llm.entity_processor import EntityProcessor
from utils.logger import logger, setup_logger
import config_neo4j as config

# Load environment variables from .env file
load_dotenv()


def get_default_concurrency():
    """
    Sensible default based on model or environment.
    Local vLLM deployments need lower concurrency than cloud APIs.
    """
    model = os.getenv('LLM_MODEL', 'gpt-4o').lower()
    # Conservative defaults for local/vLLM deployments
    if any(x in model for x in ['qwen', 'llama', 'mistral', 'local', 'deepseek']):
        return 5  # Very conservative for local models
    return 10  # Still conservative for self-hosted


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoreWeaver Knowledge Graph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Stage selection
    parser.add_argument(
        '--stage',
        action='append',
        choices=['shadow', 'spatial', 'section-map', 'entity', 'all'],
        help='Pipeline stage(s) to run (can specify multiple). Default: all'
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

    # LLM Config (from env or CLI override)
    parser.add_argument(
        '--api-key',
        default=os.getenv('OPENAI_API_KEY'),
        help='LLM API key (default: from OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--base-url',
        default=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        help='LLM base URL (default: from OPENAI_BASE_URL env var)'
    )
    parser.add_argument(
        '--model',
        default=os.getenv('LLM_MODEL', 'gpt-4o'),
        help='LLM model (default: from LLM_MODEL env var)'
    )

    # Multi-pass and concurrency options
    parser.add_argument(
        '--multi-pass',
        action='store_true',
        help='Enable multi-pass extraction mode (optimized for smaller models like qwen3-8b)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=get_default_concurrency(),
        help=f'Maximum concurrent LLM requests (default: auto-detected, use 50-100 for cloud APIs)'
    )

    # File paths
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

    return parser.parse_args()


class Pipeline:
    """LoreWeaver pipeline for processing D&D adventures into knowledge graphs."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging with experiment name from output directory
        exp_name = self.output_dir.name
        setup_logger(exp_name=exp_name)

        # File paths
        self.input_file = Path(args.input)
        self.shadow_file = self.output_dir / "shadow_tree.json"
        self.intermediate_file = self.output_dir / "shadow_tree_with_spatial_summary.json"
        self.location_graph_file = self.output_dir / "location_graph.json"
        self.section_location_map_file = self.output_dir / "section_location_map.json"
        self.entity_graph_file = self.output_dir / "entity_graph.json"

        self.shadow_tree = None

        # Validate API key
        if not self.args.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY in .env file or use --api-key"
            )

        # Log multi-pass mode
        if self.args.multi_pass:
            logger.info("=== Multi-Pass Extraction Mode Enabled ===")
            logger.info("This mode is optimized for smaller models like qwen3-8b")

        # Log concurrency settings
        logger.info(f"Max concurrent requests: {self.args.max_concurrent}")
        if self.args.max_concurrent > 20:
            logger.warning("High concurrency may overload local vLLM servers!")

    def run(self):
        """Run the pipeline with specified stages."""
        stages = self.args.stage or ['all']

        if 'all' in stages:
            self._run_stage('shadow')
            self._run_stage('spatial')
            self._run_stage('section-map')
            self._run_stage('entity')
        else:
            for stage in stages:
                self._run_stage(stage)

    def _run_stage(self, stage: str):
        """Run a single pipeline stage."""
        if self.args.dry_run:
            logger.info(f"[Dry Run] Would run stage: {stage}")
            return

        stage_map = {
            'shadow': self.stage_shadow_tree,
            'spatial': self.stage_spatial_topology,
            'section-map': self.stage_section_mapping,
            'entity': self.stage_entity_extraction,
        }

        if stage in stage_map:
            logger.info("="*60)
            logger.info(f"Running stage: {stage.upper()}")
            logger.info("="*60)
            stage_map[stage]()
        else:
            logger.warning(f"Unknown stage: {stage}")

    # ========================================================================
    # Stage 1: Shadow Tree
    # ========================================================================

    def stage_shadow_tree(self):
        """Stage 1: Build shadow tree from adventure data."""
        if self.shadow_file.exists() and not self.args.force:
            logger.info(f"Found cached shadow tree: {self.shadow_file}")
            return

        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        logger.info(f"Loading data from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            raw_data = __import__('json').load(f)

        # 5eTools data is usually under 'data' key, or directly a list
        adventure_data = raw_data.get("data", []) if isinstance(raw_data, dict) else raw_data

        logger.info(f"Building Shadow Tree from {len(adventure_data)} items...")
        builder = ShadowTreeBuilder()
        self.shadow_tree = builder.build(adventure_data)

        logger.info(f"Saving to {self.shadow_file}...")
        with open(self.shadow_file, 'w', encoding='utf-8') as f:
            __import__('json').dump(self.shadow_tree, f, indent=2, ensure_ascii=False)

        logger.info(f"Done! Shadow tree has {len(self.shadow_tree)} root nodes")

    # ========================================================================
    # Stage 2: Spatial Topology
    # ========================================================================

    def stage_spatial_topology(self):
        """Stage 2: Extract spatial topology graph."""
        self._load_shadow_tree()

        skip_summary = self.intermediate_file.exists() and not self.args.force
        if skip_summary:
            logger.info("Found intermediate file, skipping summarization...")

        processor = SpatialTopologyProcessor(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            use_multi_pass=self.args.multi_pass,
            max_concurrent=self.args.max_concurrent
        )

        location_graph = processor.process(self.shadow_tree, skip_summary=skip_summary)

        if not skip_summary:
            logger.info(f"Saving intermediate summaries to {self.intermediate_file}...")
            with open(self.intermediate_file, 'w', encoding='utf-8') as f:
                __import__('json').dump(self.shadow_tree, f, indent=2, ensure_ascii=False)

        logger.info(f"Saving location graph to {self.location_graph_file}...")
        with open(self.location_graph_file, 'w', encoding='utf-8') as f:
            __import__('json').dump(location_graph, f, indent=2, ensure_ascii=False)

        logger.info(f"Done! Location graph: {len(location_graph['nodes'])} nodes, {len(location_graph['edges'])} edges")

    # ========================================================================
    # Stage 3: Section Mapping
    # ========================================================================

    def stage_section_mapping(self):
        """Stage 3: Map sections to locations."""
        self._load_shadow_tree()

        if not self.location_graph_file.exists():
            raise FileNotFoundError(
                f"Location graph not found: {self.location_graph_file}. "
                "Run --stage spatial first"
            )

        with open(self.location_graph_file, 'r', encoding='utf-8') as f:
            location_graph = __import__('json').load(f)

        logger.info(f"Location graph: {len(location_graph['nodes'])} nodes, {len(location_graph['edges'])} edges")

        mapper = SectionLocationMapper(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent
        )

        section_map = mapper.process(self.shadow_tree, location_graph)

        logger.info(f"Saving section-location map to {self.section_location_map_file}...")
        with open(self.section_location_map_file, 'w', encoding='utf-8') as f:
            __import__('json').dump(section_map, f, indent=2, ensure_ascii=False)

        logger.info(f"Done! Mapped {len(section_map)} sections")

    # ========================================================================
    # Stage 4: Entity Extraction
    # ========================================================================

    def stage_entity_extraction(self):
        """Stage 4: Extract entities and relations."""
        self._load_shadow_tree()

        if not self.section_location_map_file.exists():
            raise FileNotFoundError(
                f"Section map not found: {self.section_location_map_file}. "
                "Run --stage section-map first"
            )

        with open(self.section_location_map_file, 'r', encoding='utf-8') as f:
            section_map = __import__('json').load(f)

        logger.info(f"Section map: {len(section_map)} sections")

        processor = EntityProcessor(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent
        )

        entity_graph = processor.process(self.shadow_tree, section_map)

        logger.info(f"Saving entity graph to {self.entity_graph_file}...")
        with open(self.entity_graph_file, 'w', encoding='utf-8') as f:
            __import__('json').dump(entity_graph, f, indent=2, ensure_ascii=False)

        logger.info(f"Done! Entity graph: {len(entity_graph['nodes'])} nodes, {len(entity_graph['edges'])} edges")

        # Print statistics
        type_counts = Counter(n.get("type", "Unknown").title() for n in entity_graph["nodes"])
        logger.info("Entity type distribution:")
        for node_type, count in type_counts.most_common():
            logger.info(f"  {node_type}: {count}")

    # ========================================================================
    # Helpers
    # ========================================================================

    def _load_shadow_tree(self):
        """Load shadow tree from file (cache or intermediate)."""
        if not self.shadow_file.exists():
            raise FileNotFoundError(
                f"Shadow tree not found: {self.shadow_file}. "
                "Run --stage shadow first"
            )

        with open(self.shadow_file, 'r', encoding='utf-8') as f:
            self.shadow_tree = __import__('json').load(f)

        # Load intermediate if available (has spatial summaries)
        if self.intermediate_file.exists():
            with open(self.intermediate_file, 'r', encoding='utf-8') as f:
                self.shadow_tree = __import__('json').load(f)
            logger.info("Using shadow tree with spatial summaries")


def main():
    """Main entry point."""
    args = parse_args()
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
