"""
Debug script to trace LLM extraction for a specific node ID.

This script reproduces the extraction process for a single node, showing:
- The content sent to the LLM
- The prompt used
- The raw LLM response
- The parsed result

Usage:
    cd /path/to/LoreWeaver
    python -m scripts.evaluation.debug_extraction --node-id 01e --parsed output/qwen3/adventure_parsed.json
"""
# ruff: noqa: E402 - Module level imports not at top (required for sys.path modification)
import sys
from pathlib import Path

# Add project root to path for imports (must be before project imports)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv

from llm.small_model_processor import SmallModelProcessor
from utils.logger import logger, setup_logger

load_dotenv()


def find_node(parsed_data: dict, node_id: str) -> dict | None:
    """Find a node by ID in the parsed adventure data."""
    internal_index = parsed_data.get("internal_index", {})
    return internal_index.get(node_id)


def build_content_tree_for_node(node: dict) -> list[dict]:
    """Build a minimal content tree containing just this node."""
    return [{
        "id": node.get("id"),
        "title": node.get("name", "Untitled"),
        "type": node.get("type", "section"),
        "content": "\n\n".join(node.get("text_content", [])),
        "children": []
    }]


async def debug_extraction(
    node: dict,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 8192
):
    """Debug extraction for a single node."""

    node_id = node.get("id")
    title = node.get("name", "Untitled")
    content = "\n\n".join(node.get("text_content", []))

    logger.info("=" * 80)
    logger.info(f"DEBUGGING EXTRACTION FOR NODE: {node_id}")
    logger.info("=" * 80)

    logger.info(f"\nNode: {node_id}")
    logger.info(f"Name: {title}")
    logger.info(f"Type: {node.get('type')}")
    logger.info(f"Parent: {node.get('parent_id')}")
    logger.info(f"Content length: {len(content)} chars")

    # Show content preview
    logger.info(f"\n{'='*80}")
    logger.info("CONTENT SENT TO LLM:")
    logger.info(f"{'='*80}")
    if len(content) > 1000:
        logger.info(content[:1000] + "\n... [truncated] ...\n")
        logger.info(f"\n[Total content: {len(content)} characters]")
    else:
        logger.info(content)

    # Initialize processor
    processor = SmallModelProcessor(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_concurrent=1,
        output_dir=None,
        use_natural_language=True,
        max_tokens=max_tokens
    )

    # Get the prompt
    from llm.prompt_factory import PromptFactory

    prompt = PromptFactory.create_unified_extraction_prompt_natural(
        title=title,
        content=content[:3000],  # Truncate like the actual processor does
        parent_context="",
        known_entities=""
    )

    logger.info(f"\n{'='*80}")
    logger.info("PROMPT SENT TO LLM:")
    logger.info(f"{'='*80}")
    logger.info(prompt[:2000] + "\n... [truncated] ..." if len(prompt) > 2000 else prompt)

    # Make the LLM call
    logger.info(f"\n{'='*80}")
    logger.info("CALLING LLM...")
    logger.info(f"{'='*80}")

    try:
        raw_response = await processor._call_llm_async(
            prompt,
            temperature=0.75,
            top_p=0.90,
            max_tokens=max_tokens,
            enable_thinking=False,
            stop=None  # Don't stop early - let LLM complete the response
        )

        logger.info(f"\n{'='*80}")
        logger.info("RAW LLM RESPONSE:")
        logger.info(f"{'='*80}")
        logger.info(raw_response)

        # Parse the response
        from llm.natural_parsers import parse_unified_extraction
        result = parse_unified_extraction(raw_response)

        logger.info(f"\n{'='*80}")
        logger.info("PARSED RESULT:")
        logger.info(f"{'='*80}")

        entities = result.get("entities", [])
        events = result.get("events", [])
        summary = result.get("summary")

        if summary:
            logger.info(f"Summary: {summary}")
        else:
            logger.info(f"Entities extracted: {len(entities)}")
            for i, e in enumerate(entities, 1):
                logger.info(f"  {i}. {e}")

            logger.info(f"\nEvents extracted: {len(events)}")
            for i, e in enumerate(events, 1):
                logger.info(f"  {i}. {e}")

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await processor.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug LLM extraction for a specific node"
    )
    parser.add_argument(
        '--node-id',
        required=True,
        help='Node ID to debug (e.g., 01e)'
    )
    parser.add_argument(
        '--parsed',
        default='output/qwen3/adventure_parsed.json',
        help='Path to parsed adventure JSON'
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('OPENAI_API_KEY'),
        help='LLM API key'
    )
    parser.add_argument(
        '--base-url',
        default=os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com/v1'),
        help='LLM base URL'
    )
    parser.add_argument(
        '--model',
        default=os.getenv('LLM_MODEL', 'qwen3-8b'),
        help='LLM model'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=int(os.getenv('LLM_MAX_TOKENS', '8192')),
        help='Max tokens for LLM response'
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if not args.api_key:
        logger.error("API key required. Set OPENAI_API_KEY in .env or use --api-key")
        return

    setup_logger(exp_name="debug_extraction")

    # Load parsed data
    logger.info(f"Loading parsed data from {args.parsed}...")
    with open(args.parsed, encoding='utf-8') as f:
        parsed_data = json.load(f)

    # Find the node
    node = find_node(parsed_data, args.node_id)
    if not node:
        logger.error(f"Node {args.node_id} not found in parsed data")
        return

    # Debug extraction (processor is created and closed inside the function)
    await debug_extraction(
        node,
        args.api_key,
        args.base_url,
        args.model,
        args.max_tokens
    )


if __name__ == "__main__":
    asyncio.run(main())
