"""
Simple regex-based entity detection test for JSON files.

Reads a list of expected entities from a text file (one per line)
and checks if they appear in a JSON file using regex.

Usage:
    python tests/test_entity_detection.py <json_file> <entities_file>

Example:
    python tests/test_entity_detection.py output/deepseek/entity_graph.json tests/samples/Stormwreck_isle.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path


def load_entities_from_file(file_path: str) -> list[str]:
    """
    Load entity names from a text file (one per line).

    Args:
        file_path: Path to text file with entity names

    Returns:
        List of entity names (non-empty lines)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entities = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            entities.append(line)

    return entities


def find_entities_in_json(file_path: str, search_terms: list[str]) -> dict[str, list[str]]:
    """
    Search for entity names in a JSON file using regex.

    Args:
        file_path: Path to JSON file
        search_terms: List of entity names to search for

    Returns:
        Dict mapping each search term to list of matches found
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    results = {}

    for term in search_terms:
        # Case-insensitive regex search
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = pattern.findall(content)

        if matches:
            results[term] = matches

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test if entities from a list file exist in a JSON file using regex"
    )
    parser.add_argument(
        "json_file",
        help="JSON file to search in (e.g., output/deepseek/entity_graph.json)"
    )
    parser.add_argument(
        "entities_file",
        help="Text file with entity names (one per line, e.g., tests/samples/Stormwreck_isle.txt)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed match context"
    )

    args = parser.parse_args()

    json_path = Path(args.json_file)
    entities_path = Path(args.entities_file)

    if not json_path.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)

    if not entities_path.exists():
        print(f"Error: Entities file not found: {args.entities_file}")
        sys.exit(1)

    # Load expected entities from file
    expected_entities = load_entities_from_file(str(entities_path))

    print(f"\n=== Checking {len(expected_entities)} entities in {args.json_file} ===\n")
    print(f"Expected entities from: {args.entities_file}\n")

    # Search for entities
    results = find_entities_in_json(str(json_path), expected_entities)

    found_count = 0
    missing = []

    for term in expected_entities:
        matches = results.get(term, [])
        if matches:
            found_count += 1
            print(f"[+] FOUND: '{term}' - {len(matches)} occurrence(s)")
            if args.verbose:
                # Show some context around matches
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                for match in pattern.finditer(content):
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 30)
                    context = content[start:end].replace('\n', ' ')
                    print(f"  ...{context}...")
        else:
            missing.append(term)
            print(f"[-] MISSING: '{term}'")

    print(f"\nSummary: {found_count}/{len(expected_entities)} entities found")

    if missing:
        print(f"\nMissing entities ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("\nAll entities found!")


if __name__ == "__main__":
    main()
