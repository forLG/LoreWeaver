"""
Evaluate knowledge graph extraction quality against human labels.

Features fuzzy matching for:
- Case insensitivity
- Whitespace normalization
- Number handling (Area1 vs Area 1)
- Partial/substring matches

Usage:
    cd /path/to/LoreWeaver
    python -m scripts.evaluation.evaluate_extraction --graph output/qwen3/semantic_graph.json \
                                                    --labels tests/samples/Stormwreck_Isle.txt
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import logger, setup_logger


def normalize_name(name: str) -> str:
    """
    Normalize entity name for fuzzy matching.

    Handles:
    - Case conversion
    - Whitespace collapsing
    - Apostrophe/quote removal
    - Special character normalization
    """
    if not name:
        return ""

    # Convert to lowercase
    name = name.lower()

    # Remove apostrophes and quotes
    name = name.replace("'", "").replace('"', "").replace("`", "")

    # Collapse multiple spaces to single space
    name = re.sub(r'\s+', ' ', name)

    # Remove leading/trailing whitespace
    name = name.strip()

    return name


def fuzzy_match_entities(
    human_entities: set[str],
    extracted_entities: set[str],
    threshold: int = 80
) -> dict[str, Any]:
    """
    Match human labels with extracted entities using fuzzy matching.

    Args:
        human_entities: Set of human-labeled entity names
        extracted_entities: Set of extracted entity names
        threshold: Minimum similarity score (0-100) for match

    Returns:
        Dict with matches, missing, extra, and match details
    """
    human_list = sorted(human_entities)
    extracted_list = sorted(extracted_entities)

    # Normalize names for matching
    human_normalized = {normalize_name(e): e for e in human_list}
    extracted_normalized = {normalize_name(e): e for e in extracted_list}

    matched_human = set()
    matched_extracted = set()
    match_details = []

    # First pass: exact normalized matches
    for norm_human, human_name in human_normalized.items():
        if norm_human in extracted_normalized:
            extracted_name = extracted_normalized[norm_human]
            matched_human.add(human_name)
            matched_extracted.add(extracted_name)
            match_details.append({
                "human": human_name,
                "extracted": extracted_name,
                "score": 100,
                "method": "exact_normalized"
            })

    # Second pass: fuzzy match remaining entities
    remaining_human = [e for e in human_list if e not in matched_human]
    remaining_extracted = [e for e in extracted_list if e not in matched_extracted]

    for human_name in remaining_human:
        norm_human = normalize_name(human_name)

        # Use rapidfuzz for fuzzy matching
        result = process.extractOne(
            norm_human,
            [normalize_name(e) for e in remaining_extracted],
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )

        if result:
            match_norm, score, _ = result
            # Find the original name
            extracted_name = extracted_normalized.get(match_norm)
            if extracted_name and extracted_name not in matched_extracted:
                matched_human.add(human_name)
                matched_extracted.add(extracted_name)
                match_details.append({
                    "human": human_name,
                    "extracted": extracted_name,
                    "score": score,
                    "method": "fuzzy"
                })

    # Calculate stats
    missing = [e for e in human_list if e not in matched_human]
    extra = [e for e in extracted_list if e not in matched_extracted]

    return {
        "matches": len(match_details),
        "match_details": match_details,
        "missing": missing,
        "extra": extra,
        "human_count": len(human_list),
        "extracted_count": len(extracted_list),
    }


def calculate_metrics(human_count: int, extracted_count: int, matches: int) -> dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    precision = matches / extracted_count if extracted_count > 0 else 0
    recall = matches / human_count if human_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }


def load_human_labels(label_file: str) -> dict[str, set[str]]:
    """
    Load human-labeled entities from text file.

    Expected format (one entity per line):
        Entity Name

    Note: Labels may be incomplete and have formatting issues.
    """
    entities = {
        "all": set(),
        "creatures": set(),
        "locations": set(),
        "items": set(),
        "unknown": set(),
    }

    with open(label_file, encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith('#'):  # Skip empty lines and comments
                continue

            entities["all"].add(name)

            # Heuristic classification (will be refined manually)
            name_lower = name.lower()

            # Creature indicators
            if any(kw in name_lower for kw in [
                'dragon', 'zombie', 'kobold', 'harpy', 'ghoul', 'owlbear',
                'fungi', 'stirge', 'snake', 'spore', 'servant', 'octopus',
                'sailor', 'sharruth', 'runara', 'agga', 'blepp', 'frub',
                'kilnip', 'laylee', 'mumpo', 'myla', 'rix', 'zark', 'tarak',
                'varnoth', 'aidron', 'sinensa', 'bispo', 'valup', 'popple',
                'orcus', 'tides', 'renegade'
            ]):
                entities["creatures"].add(name)

            # Location indicators
            elif any(kw in name_lower for kw in [
                'rest', 'house', 'kitchen', 'library', 'temple', 'cave',
                'tunnel', 'farm', 'chamber', 'larder', 'deck', 'nest',
                'castle', 'quarters', 'hall', 'hold', 'observatory',
                'anchor', 'camp', 'study', 'tower', 'entrance', 'isle'
            ]):
                entities["locations"].add(name)

            # Item indicators
            elif any(kw in name_lower for kw in [
                'treasure', 'sculpture', 'effigie', 'crystal', 'bridge'
            ]):
                entities["items"].add(name)

            else:
                entities["unknown"].add(name)

    return entities


def load_extracted_entities(graph_file: str) -> dict[str, set[str]]:
    """Load extracted entities from semantic graph JSON."""
    with open(graph_file, encoding='utf-8') as f:
        graph = json.load(f)

    entities = {
        "all": set(),
        "creatures": set(),
        "locations": set(),
        "items": set(),
        "events": set(),
    }

    for node in graph.get("nodes", []):
        if node.get("node_type") == "event":
            entities["events"].add(node.get("label", ""))
            continue

        label = node.get("label", "")
        if not label:
            continue

        entities["all"].add(label)

        etype = node.get("type", "")
        if etype == "Creature":
            entities["creatures"].add(label)
        elif etype == "Location":
            entities["locations"].add(label)
        elif etype == "Item":
            entities["items"].add(label)

    return entities


def print_evaluation(human_entities: dict, extracted_entities: dict, threshold: int = 80):
    """Print evaluation results."""

    logger.info("=" * 70)
    logger.info("ENTITY EXTRACTION EVALUATION (Fuzzy Matching)")
    logger.info(f"Fuzzy Match Threshold: {threshold}%")
    logger.info("=" * 70)

    # Overall evaluation
    result = fuzzy_match_entities(human_entities["all"], extracted_entities["all"], threshold)
    metrics = calculate_metrics(result["human_count"], result["extracted_count"], result["matches"])

    logger.info(f"\nOverall:")
    logger.info(f"  Human labels: {result['human_count']}")
    logger.info(f"  Extracted: {result['extracted_count']}")
    logger.info(f"  Matches: {result['matches']}")
    logger.info(f"  Missing: {len(result['missing'])}")
    logger.info(f"  Extra: {len(result['extra'])}")

    logger.info(f"\nMetrics:")
    logger.info(f"  Precision: {metrics['precision']:.1f}%")
    logger.info(f"  Recall:    {metrics['recall']:.1f}%")
    logger.info(f"  F1 Score:  {metrics['f1']:.1f}%")

    # By category
    categories = ["creatures", "locations", "items"]

    for category in categories:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{category.upper()}")
        logger.info(f"{'=' * 60}")

        h_cat = human_entities[category]
        e_cat = extracted_entities[category]

        if not h_cat and not e_cat:
            logger.info("  No entities in this category")
            continue

        result = fuzzy_match_entities(h_cat, e_cat, threshold)
        metrics = calculate_metrics(result["human_count"], result["extracted_count"], result["matches"])

        logger.info(f"\n  Human: {result['human_count']} | Extracted: {result['extracted_count']} | Matches: {result['matches']}")
        logger.info(f"  Precision: {metrics['precision']:.1f}% | Recall: {metrics['recall']:.1f}% | F1: {metrics['f1']:.1f}%")

        # Show matches with scores
        if result["match_details"]:
            logger.info(f"\n  Matched entities (showing fuzzy matches):")
            fuzzy_matches = [m for m in result["match_details"] if m["method"] == "fuzzy"]
            if fuzzy_matches:
                for m in fuzzy_matches[:5]:
                    logger.info(f"    '{m['human']}' -> '{m['extracted']}' ({m['score']:.0f}%)")
                if len(fuzzy_matches) > 5:
                    logger.info(f"    ... and {len(fuzzy_matches) - 5} more fuzzy matches")

        # Show missing (limit output)
        if result["missing"]:
            logger.info(f"\n  Missing ({len(result['missing'])}):")
            for e in result["missing"][:10]:
                logger.info(f"    - {e}")
            if len(result["missing"]) > 10:
                logger.info(f"    ... and {len(result['missing']) - 10} more")

        # Show extra (limit output)
        if result["extra"]:
            logger.info(f"\n  Extra/Hallucinations ({len(result['extra'])}):")
            for e in result["extra"][:10]:
                logger.info(f"    - {e}")
            if len(result["extra"]) > 10:
                logger.info(f"    ... and {len(result['extra']) - 10} more")

    # Events
    if extracted_entities["events"]:
        logger.info(f"\n{'=' * 60}")
        logger.info("EVENTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Extracted {len(extracted_entities['events'])} events (no human labels for comparison)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge graph extraction quality with fuzzy matching"
    )
    parser.add_argument(
        '--graph',
        default='output/qwen3/semantic_graph.json',
        help='Path to semantic graph JSON file'
    )
    parser.add_argument(
        '--labels',
        default='tests/samples/Stormwreck_Isle.txt',
        help='Path to human-labeled entities file'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=80,
        help='Fuzzy match threshold (0-100, default: 80)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logger(exp_name="evaluation")

    logger.info(f"Loading human labels from {args.labels}...")
    logger.info("Note: Human labels are incomplete and may have formatting issues")
    human_entities = load_human_labels(args.labels)
    logger.info(f"  Loaded {len(human_entities['all'])} human labels")
    logger.info(f"    Creatures: {len(human_entities['creatures'])}")
    logger.info(f"    Locations: {len(human_entities['locations'])}")
    logger.info(f"    Items: {len(human_entities['items'])}")

    logger.info(f"\nLoading extracted entities from {args.graph}...")
    extracted_entities = load_extracted_entities(args.graph)
    logger.info(f"  Loaded {len(extracted_entities['all'])} extracted entities")
    logger.info(f"    Creatures: {len(extracted_entities['creatures'])}")
    logger.info(f"    Locations: {len(extracted_entities['locations'])}")
    logger.info(f"    Items: {len(extracted_entities['items'])}")
    logger.info(f"    Events: {len(extracted_entities['events'])}")

    # Print evaluation
    print_evaluation(human_entities, extracted_entities, args.threshold)


if __name__ == "__main__":
    main()
