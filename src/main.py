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

    # Use specific model (processor auto-selected based on model capability)
    python -m main --stage all --model qwen3-8b

    # Adjust concurrency for local vLLM deployments
    python -m main --stage all --max-concurrent 5

Environment Variables (.env):
    OPENAI_API_KEY     - Your LLM API key (required)
    OPENAI_BASE_URL    - API base URL (default: https://api.openai.com/v1)
    LLM_MODEL          - Model name (default: gpt-4o)
                        Small models (qwen, llama, mistral): Entity-first pipeline
                        Large models (gpt-4o, deepseek): Spatial-first pipeline
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

import config_neo4j as config
from builder.shadow_builder import ShadowTreeBuilder
from llm.entity_processor import EntityProcessor
from llm.spatial_processor import SectionLocationMapper, SpatialTopologyProcessor
from utils.logger import logger, setup_logger

# Load environment variables from .env file
load_dotenv()


def get_default_concurrency():
    """
    Sensible default based on model or environment.
    Local vLLM deployments need lower concurrency than cloud APIs.
    """
    model = os.getenv('LLM_MODEL', 'deepseek').lower()
    env_concurrency = os.getenv('LLM_MAX_CONCURRENT')
    logger.debug(f"Auto-detecting concurrency for model: {model}")
    logger.debug(f"Environment concurrency override: {env_concurrency}")
    if env_concurrency:
        try:
            return int(env_concurrency)
        except ValueError:
            pass
    logger.info("Using conservative defaults based on model type")
    # Conservative defaults for local/vLLM deployments
    if any(x in model for x in ['qwen', 'local']):
        # Very conservative for local models
        return 5
    return 50


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
        default=os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com/v1'),
        help='LLM base URL (default: from OPENAI_BASE_URL env var)'
    )
    parser.add_argument(
        '--model',
        default=os.getenv('LLM_MODEL', 'deepseek-chat'),
        help='LLM model (default: from LLM_MODEL env var)'
    )

    # Concurrency options
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=get_default_concurrency(),
        help='Maximum concurrent LLM requests (default: auto-detected, use 50-100 for cloud APIs)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=int(os.getenv('LLM_MAX_TOKENS', '0')),
        help='Maximum tokens per LLM response (default: from LLM_MAX_TOKENS env var, 0 = no limit)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=float(os.getenv('LLM_REPETITION_PENALTY', '0')) if os.getenv('LLM_REPETITION_PENALTY') else None,
        help='Repetition penalty for vLLM (default: from LLM_REPETITION_PENALTY env var, 1.0 = no penalty)'
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

        # Log model and processor selection
        is_small_model = any(x in self.args.model.lower() for x in ['qwen', 'local'])
        logger.info(f"Model: {self.args.model}")
        if is_small_model:
            logger.info("Processor: SmallModelProcessor (entity-first pipeline)")
        else:
            logger.info("Processor: SpatialTopologyProcessor (spatial-first pipeline)")

        # Log concurrency settings
        logger.info(f"Max concurrent requests: {self.args.max_concurrent}")
        if self.args.max_concurrent > 20:
            logger.warning("High concurrency may overload local vLLM servers!")

    def run(self):
        """Run the pipeline with specified stages."""
        stages = self.args.stage or ['all']
        is_small_model = any(x in self.args.model.lower() for x in ['qwen', 'local'])

        if 'all' in stages:
            self._run_stage('shadow')
            self._run_stage('spatial')
            if is_small_model:
                # Small models use unified pipeline, skip these stages
                # historically we call it "spatial stage"
                logger.info("Skipping section-map and entity stages for small model pipeline")
                return
            self._run_stage('section-map')
            self._run_stage('entity')
        else:
            for stage in stages:
                if is_small_model and stage in ['section-map', 'entity']:
                    logger.info(f"Skipping stage {stage} for small model pipeline")
                    continue
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

        # 5eTools data is usually under 'data' key, or directly a list
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

        # Choose processor based on model size
        is_small_model = any(x in self.args.model.lower() for x in ['qwen', 'local'])

        if is_small_model:
            # Use unified entity+event pipeline for small models
            from llm.small_model_processor import SmallModelProcessor
            processor = SmallModelProcessor(
                api_key=self.args.api_key,
                base_url=self.args.base_url,
                model=self.args.model,
                max_concurrent=self.args.max_concurrent,
                output_dir=self.output_dir,  # Save debug outputs
                max_tokens=self.args.max_tokens,
                repetition_penalty=self.args.repetition_penalty
            )
            location_graph = processor.process(self.shadow_tree, skip_summary=False)
        else:
            # Use standard spatial processor for large models
            processor = SpatialTopologyProcessor(
                api_key=self.args.api_key,
                base_url=self.args.base_url,
                model=self.args.model,
                max_concurrent=self.args.max_concurrent,
                repetition_penalty=self.args.repetition_penalty
            )
            location_graph = processor.process(self.shadow_tree, skip_summary=skip_summary)

        if not skip_summary and not is_small_model:
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
            repetition_penalty=self.args.repetition_penalty
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
            repetition_penalty=self.args.repetition_penalty
        )

        entity_graph = processor.process(self.shadow_tree, section_map)

        logger.info(f"Saving entity graph to {self.entity_graph_file}...")
        with open(self.entity_graph_file, 'w', encoding='utf-8') as f:
            json.dump(entity_graph.to_dict() if hasattr(entity_graph, 'to_dict') else entity_graph, f, indent=2, ensure_ascii=False)

        # Convert to dict for accessing nodes/edges
        graph_dict = entity_graph.to_dict() if hasattr(entity_graph, 'to_dict') else entity_graph

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
        """Load shadow tree from file (cache or intermediate)."""
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
