"""
Side-by-side comparison of human labels vs extracted entities.

Shows detailed matches, near-misses, and analysis of extraction quality.

Usage:
    cd /path/to/LoreWeaver
    python -m scripts.evaluation.compare_entities --graph output/qwen3/semantic_graph.json \
                                                --labels tests/samples/Stormwreck_Isle.txt
"""
import argparse
import json
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import logger, setup_logger


def normalize_name(name: str) -> str:
    """Normalize entity name for comparison."""
    if not name:
        return ""
    name = name.lower().replace("'", "").replace('"', "").replace("`", "")
    name = name.replace('_', ' ')
    name = ' '.join(name.split())
    return name.strip()


def load_human_labels(label_file: str) -> list[dict]:
    """Load human labels with metadata."""
    entities = []
    with open(label_file, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            name = line.strip()
            if not name or name.startswith('#'):
                continue
            entities.append({
                "name": name,
                "line": i,
                "normalized": normalize_name(name),
                "source": "human"
            })
    return entities


def load_extracted_entities(graph_file: str) -> list[dict]:
    """Load extracted entities with metadata."""
    with open(graph_file, encoding='utf-8') as f:
        graph = json.load(f)

    entities = []
    for node in graph.get("nodes", []):
        if node.get("node_type") == "event":
            continue

        label = node.get("label", "")
        if not label:
            continue

        entities.append({
            "name": label,
            "type": node.get("type", "Unknown"),
            "normalized": normalize_name(label),
            "source": "extracted",
            "id": node.get("id", ""),
            "extraction_method": node.get("extraction_method", "unknown"),
            "is_generic": node.get("is_generic", False),
            "location_type": node.get("location_type", ""),
            "creature_type": node.get("creature_type", ""),
        })
    return entities


def find_best_match(human_entity: dict, extracted_entities: list[dict], threshold: int = 70) -> dict | None:
    """Find best matching extracted entity for a human label."""
    result = process.extractOne(
        human_entity["normalized"],
        [e["normalized"] for e in extracted_entities],
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )

    if result:
        match_norm, score, _ = result
        for e in extracted_entities:
            if e["normalized"] == match_norm:
                return {
                    "extracted": e,
                    "score": score
                }
    return None


def print_comparison(human_entities: list[dict], extracted_entities: list[dict], threshold: int = 70):
    """Print detailed side-by-side comparison."""

    logger.info("=" * 100)
    logger.info("SIDE-BY-SIDE ENTITY COMPARISON")
    logger.info(f"Threshold: {threshold}% similarity required for match")
    logger.info("=" * 100)

    # Categorize matches
    exact_matches = []
    high_confidence = []
    medium_confidence = []
    low_confidence = []
    no_match = []

    matched_extracted = set()

    for h in human_entities:
        match = find_best_match(h, extracted_entities, threshold)

        if match:
            e = match["extracted"]
            matched_extracted.add(e["id"])
            score = match["score"]

            if score == 100:
                exact_matches.append((h, e, score))
            elif score >= 90:
                high_confidence.append((h, e, score))
            elif score >= 80:
                medium_confidence.append((h, e, score))
            else:
                low_confidence.append((h, e, score))
        else:
            no_match.append(h)

    # Unmatched extracted entities
    unmatched_extracted = [e for e in extracted_entities if e["id"] not in matched_extracted]

    # Print summary
    logger.info(f"\nSUMMARY:")
    logger.info(f"  Human labels: {len(human_entities)}")
    logger.info(f"  Extracted: {len(extracted_entities)}")
    logger.info(f"  Exact matches (100%): {len(exact_matches)}")
    logger.info(f"  High confidence (90-99%): {len(high_confidence)}")
    logger.info(f"  Medium confidence (80-89%): {len(medium_confidence)}")
    logger.info(f"  Low confidence (70-79%): {len(low_confidence)}")
    logger.info(f"  No match found: {len(no_match)}")
    logger.info(f"  Extra extracted (not in human): {len(unmatched_extracted)}")

    # Print exact matches
    if exact_matches:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"EXACT MATCHES (100% similarity) - {len(exact_matches)} entities")
        logger.info(f"{'=' * 100}")
        logger.info(f"{'Human Label':<40} | {'Extracted':<40} | {'Type':<15}")
        logger.info("-" * 100)
        for h, e, _ in exact_matches:
            logger.info(f"{h['name']:<40} | {e['name']:<40} | {e['type']:<15}")

    # Print high confidence matches
    if high_confidence:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"HIGH CONFIDENCE MATCHES (90-99% similarity) - {len(high_confidence)} entities")
        logger.info(f"{'=' * 100}")
        logger.info(f"{'Human Label':<40} | {'Extracted':<40} | {'Score':<10} | {'Type':<15}")
        logger.info("-" * 100)
        for h, e, score in high_confidence:
            logger.info(f"{h['name']:<40} | {e['name']:<40} | {score:>3}%       | {e['type']:<15}")

    # Print medium confidence matches
    if medium_confidence:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"MEDIUM CONFIDENCE MATCHES (80-89% similarity) - {len(medium_confidence)} entities")
        logger.info(f"{'=' * 100}")
        logger.info(f"{'Human Label':<40} | {'Extracted':<40} | {'Score':<10} | {'Type':<15}")
        logger.info("-" * 100)
        for h, e, score in medium_confidence:
            logger.info(f"{h['name']:<40} | {e['name']:<40} | {score:>3}%       | {e['type']:<15}")

    # Print low confidence matches (potential issues)
    if low_confidence:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"LOW CONFIDENCE MATCHES (70-79% similarity) - {len(low_confidence)} entities")
        logger.info(f"{'=' * 100}")
        logger.info(f"{'Human Label':<40} | {'Extracted':<40} | {'Score':<10} | {'Type':<15}")
        logger.info("-" * 100)
        for h, e, score in low_confidence:
            logger.info(f"{h['name']:<40} | {e['name']:<40} | {score:>3}%       | {e['type']:<15}")

    # Print missing entities (no match found)
    if no_match:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"MISSING ENTITIES (no match found) - {len(no_match)} entities")
        logger.info(f"{'=' * 100}")
        logger.info(f"{'Human Label':<40} | {'Line':<10}")
        logger.info("-" * 100)
        for h in no_match[:50]:  # Limit to 50
            logger.info(f"{h['name']:<40} | {h['line']:<10}")
        if len(no_match) > 50:
            logger.info(f"... and {len(no_match) - 50} more")

    # Print extra extracted entities
    if unmatched_extracted:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"EXTRA EXTRACTED ENTITIES (not in human labels) - {len(unmatched_extracted)} entities")
        logger.info(f"{'=' * 100}")
        logger.info(f"{'Extracted':<40} | {'Type':<15} | {'Method':<15}")
        logger.info("-" * 100)

        # Group by type
        by_type = {}
        for e in unmatched_extracted:
            etype = e.get("type", "Unknown")
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(e)

        for etype, entities in sorted(by_type.items()):
            logger.info(f"\n{etype} ({len(entities)}):")
            for e in entities[:30]:  # Limit to 30 per type
                method = e.get("extraction_method", "unknown")
                generic = " (generic)" if e.get("is_generic") else ""
                logger.info(f"  {e['name']:<40} | {e['type']:<15} | {method}{generic}")
            if len(entities) > 30:
                logger.info(f"  ... and {len(entities) - 30} more {etype.lower()} entities")


def analyze_by_type(human_entities: list[dict], extracted_entities: list[dict], threshold: int = 70):
    """Analyze matches by entity type."""

    logger.info(f"\n{'=' * 100}")
    logger.info("DETAILED ANALYSIS BY TYPE")
    logger.info(f"{'=' * 100}")

    # Group extracted by type
    by_type = {}
    for e in extracted_entities:
        etype = e.get("type", "Unknown")
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(e)

    # For each human label, find if it matched and to what type
    type_matches = {}

    for h in human_entities:
        match = find_best_match(h, extracted_entities, threshold)
        if match:
            e = match["extracted"]
            etype = e.get("type", "Unknown")
            if etype not in type_matches:
                type_matches[etype] = {"matched": [], "count": 0}
            type_matches[etype]["matched"].append((h, e, match["score"]))
            type_matches[etype]["count"] += 1

    # Print per type
    for etype in ["Creature", "Location", "Item", "Group", "Unknown"]:
        extracted = by_type.get(etype, [])
        matches = type_matches.get(etype, {})
        matched_count = matches.get("count", 0)

        if not extracted and matched_count == 0:
            continue

        logger.info(f"\n{etype.upper()}:")
        logger.info(f"  Extracted: {len(extracted)} entities")
        logger.info(f"  Matched to human labels: {matched_count} entities")

        if matched_count > 0:
            logger.info(f"  Match details:")
            for h, e, score in matches["matched"][:10]:
                logger.info(f"    '{h['name']}' -> '{e['name']}' ({score}%)")
            if len(matches["matched"]) > 10:
                logger.info(f"    ... and {len(matches['matched']) - 10} more")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of human labels vs extracted entities"
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
        default=70,
        help='Match threshold (default: 70)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    setup_logger(exp_name="compare")

    logger.info(f"Loading human labels from {args.labels}...")
    human_entities = load_human_labels(args.labels)
    logger.info(f"  Loaded {len(human_entities)} human labels")

    logger.info(f"\nLoading extracted entities from {args.graph}...")
    extracted_entities = load_extracted_entities(args.graph)
    logger.info(f"  Loaded {len(extracted_entities)} extracted entities")

    # Print comparison
    print_comparison(human_entities, extracted_entities, args.threshold)

    # Analyze by type
    analyze_by_type(human_entities, extracted_entities, args.threshold)


if __name__ == "__main__":
    main()
