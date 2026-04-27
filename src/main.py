"""
LoreWeaver Main Pipeline (Large Model / Spatial-First)

Usage:
    # Run all stages
    python -m src.main --stage all

    # Run specific stages
    python -m src.main --stage shadow
    python -m src.main --stage spatial
    python -m src.main --stage section-map
    python -m src.main --stage entity

    # Run multiple stages
    python -m src.main --stage spatial --stage section-map

    # Rerun all stages, ignore cache
    python -m src.main --stage all --force

    # Preview what would run
    python -m src.main --stage all --dry-run

    # Use specific model
    python -m src.main --stage all --model gpt-4o

    # Adjust concurrency for local vLLM deployments
    python -m src.main --stage all --max-concurrent 50

Environment Variables (.env):
    OPENAI_API_KEY     - Your LLM API key (required)
    OPENAI_BASE_URL    - API base URL (default: https://api.openai.com/v1)
    LLM_MODEL          - Model name (default: gpt-4o)
    LOREWEAVER_CONTEXT_WINDOW_CHARS - Approximate per-call context budget
    LOREWEAVER_LOCATION_BATCH_SIZE  - Number of location IDs per mapping call

Pipeline Stages:
    shadow       - Build shadow tree from adventure JSON
    spatial      - Extract spatial topology graph
    section-map  - Map sections to locations
    entity       - Extract entities and relations
    all          - Run all stages
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.builder.shadow_builder import ShadowTreeBuilder
from src.llm.entity_processor import EntityProcessor
from src.llm.spatial_processor import SectionLocationMapper, SpatialTopologyProcessor
from src.utils.logger import logger, setup_logger

# Load environment variables from .env file
load_dotenv()


def get_default_concurrency():
    """
    Sensible default based on environment.
    """
    env_concurrency = os.getenv('LLM_MAX_CONCURRENT')
    if env_concurrency:
        try:
            return int(env_concurrency)
        except ValueError:
            pass
    # Default for cloud APIs
    return 50


def get_int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable with a fallback."""
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoreWeaver Knowledge Graph Pipeline (Large Model)",
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

    # LLM Config
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

    # Concurrency options
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=get_default_concurrency(),
        help='Maximum concurrent LLM requests (default: auto-detected)'
    )
    parser.add_argument(
        '--context-window-chars',
        type=int,
        default=get_int_env('LOREWEAVER_CONTEXT_WINDOW_CHARS', 24000),
        help='Approximate character budget for each long-context LLM call'
    )
    parser.add_argument(
        '--location-batch-size',
        type=int,
        default=get_int_env('LOREWEAVER_LOCATION_BATCH_SIZE', 150),
        help='Maximum number of location IDs to include in one section mapping call'
    )

    # File paths
    parser.add_argument(
        '--input',
        default='data/adventure-dosi.json',
        help='Input adventure file (default: data/adventure-dosi.json)'
    )
    parser.add_argument(
        '--output-dir',
        default="output/",
        help=f'Output directory (default: output/)'
    )

    return parser.parse_args()


class Pipeline:
    """LoreWeaver spatial-first pipeline for large models."""

    def __init__(self, args):
        self.args = args

        # Setup file logging
        self.input_file = Path(args.input)
        exp_name = self.input_file.name.replace('.json', '')
        setup_logger(exp_name=exp_name)

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path("cache/") / exp_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.shadow_file = self.cache_dir / "shadow_tree.json"
        self.intermediate_file = self.cache_dir / "shadow_tree_with_spatial_summary.json"
        self.location_graph_file = self.cache_dir / "location_graph.json"
        self.section_location_map_file = self.cache_dir / "section_location_map.json"
        self.entity_graph_file = self.output_dir / (exp_name + '.json')

        self.shadow_tree = None

        # Validate API key
        if not self.args.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY in .env file or use --api-key"
            )

        # Log configuration
        logger.info(f"Model: {self.args.model}")
        logger.info(f"Max concurrent requests: {self.args.max_concurrent}")
        logger.info(f"Context window budget: {self.args.context_window_chars} chars")
        logger.info(f"Location mapping batch size: {self.args.location_batch_size}")

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
        with open(self.input_file, encoding='utf-8') as f:
            raw_data = json.load(f)

        # 5eTools data is usually under 'data' key
        adventure_data = raw_data.get("data", []) if isinstance(raw_data, dict) else raw_data

        logger.info(f"Building Shadow Tree from {len(adventure_data)} items...")
        builder = ShadowTreeBuilder()
        self.shadow_tree = builder.build(adventure_data)

        logger.info(f"Saving to {self.shadow_file}...")
        with open(self.shadow_file, 'w', encoding='utf-8') as f:
            json.dump(self.shadow_tree, f, indent=2, ensure_ascii=False)

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

        # Use spatial processor for large models
        processor = SpatialTopologyProcessor(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent,
            context_window_chars=self.args.context_window_chars,
            location_batch_size=self.args.location_batch_size,
        )
        location_graph = processor.process(self.shadow_tree, skip_summary=skip_summary)

        if not skip_summary:
            logger.info(f"Saving intermediate summaries to {self.intermediate_file}...")
            with open(self.intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(self.shadow_tree, f, indent=2, ensure_ascii=False)

        logger.info(f"Saving location graph to {self.location_graph_file}...")
        with open(self.location_graph_file, 'w', encoding='utf-8') as f:
            json.dump(location_graph, f, indent=2, ensure_ascii=False)

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

        with open(self.location_graph_file, encoding='utf-8') as f:
            location_graph = json.load(f)

        logger.info(f"Location graph: {len(location_graph['nodes'])} nodes, {len(location_graph['edges'])} edges")

        mapper = SectionLocationMapper(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent,
            context_window_chars=self.args.context_window_chars,
            location_batch_size=self.args.location_batch_size,
        )

        section_map = mapper.process(self.shadow_tree, location_graph)

        logger.info(f"Saving section-location map to {self.section_location_map_file}...")
        with open(self.section_location_map_file, 'w', encoding='utf-8') as f:
            json.dump(section_map, f, indent=2, ensure_ascii=False)

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

        with open(self.section_location_map_file, encoding='utf-8') as f:
            section_map = json.load(f)

        logger.info(f"Section map: {len(section_map)} sections")

        processor = EntityProcessor(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            max_concurrent=self.args.max_concurrent,
            context_window_chars=self.args.context_window_chars,
        )

        entity_graph = processor.process(self.shadow_tree, section_map)

        logger.info(f"Saving entity graph to {self.entity_graph_file}...")
        with open(self.entity_graph_file, 'w', encoding='utf-8') as f:
            json.dump(entity_graph, f, indent=2, ensure_ascii=False)

        graph_dict = entity_graph
        logger.info(f"Done! Entity graph: {len(graph_dict['nodes'])} nodes, {len(graph_dict['edges'])} edges")

        # Print statistics
        type_counts = Counter(n.get("type", "Unknown").title() for n in graph_dict["nodes"])
        logger.info("Entity type distribution:")
        for node_type, count in type_counts.most_common():
            logger.info(f"  {node_type}: {count}")

    # ========================================================================
    # Helpers
    # ========================================================================

    def _load_shadow_tree(self):
        """Load shadow tree from file."""
        if not self.shadow_file.exists():
            raise FileNotFoundError(
                f"Shadow tree not found: {self.shadow_file}. "
                "Run --stage shadow first"
            )

        with open(self.shadow_file, encoding='utf-8') as f:
            self.shadow_tree = json.load(f)

        # Load intermediate if available (has spatial summaries)
        if self.intermediate_file.exists():
            with open(self.intermediate_file, encoding='utf-8') as f:
                self.shadow_tree = json.load(f)
            logger.info("Using shadow tree with spatial summaries")


def main():
    """Main entry point."""
    args = parse_args()
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
