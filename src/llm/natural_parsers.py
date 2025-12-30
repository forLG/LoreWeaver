"""
Natural Language Parsers for LLM outputs.

Robust alternative to JSON parsing for small models.
Handles truncation gracefully and is more debuggable.
"""
import re
from typing import Any


def parse_ner_entities(text: str) -> dict[str, Any]:
    """
    Parse natural language entity extraction output.

    Expected format:
        Entity: Dragon's Rest
        Type: Location
        ID: dragon_s_rest
        Aliases: temple, monastery

        Entity: Runara
        Type: Creature
        ID: runara
        Aliases: bronze dragon, elder

    Returns:
        {"entities": [...]}
    """
    entities = []
    current = {}

    # Split by double newlines (entity blocks)
    blocks = re.split(r'\n\s*\n', text.strip())

    for block in blocks:
        if not block.strip():
            continue

        current = {}
        lines = block.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse key: value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if key == 'entity':
                    current['label'] = value
                elif key == 'type':
                    current['type'] = value
                elif key == 'id':
                    current['id'] = value
                elif key == 'aliases':
                    # Split by comma, strip whitespace
                    aliases = [a.strip() for a in value.split(',') if a.strip()]
                    current['aliases'] = aliases

        # Only add if we have at least a non-empty ID or label
        label = current.get('label', '').strip()
        eid = current.get('id', '').strip()

        if label or eid:
            # Generate ID if missing
            if not eid:
                eid = label.lower().replace(' ', '_').replace("'", "")
                current['id'] = eid

            # Ensure type has a default
            if 'type' not in current:
                current['type'] = 'Entity'

            # Ensure aliases exists
            if 'aliases' not in current:
                current['aliases'] = []

            entities.append(current)

    return {"entities": entities}


def parse_relations(text: str) -> dict[str, Any]:
    """
    Parse natural language relation extraction output.

    Expected format:
        Relation: inhabits
        Source: runara
        Target: dragon_s_rest
        Description: Lives in the temple

        Relation: commands
        Source: goblin_boss
        Target: goblin_minion
        Description: Leads the goblins

    Returns:
        {"relations": [...]}
    """
    relations = []
    current = {}

    # Split by double newlines (relation blocks)
    blocks = re.split(r'\n\s*\n', text.strip())

    for block in blocks:
        if not block.strip():
            continue

        current = {}
        lines = block.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if key == 'relation':
                    current['relation'] = value
                elif key == 'source':
                    current['source'] = value
                elif key == 'target':
                    current['target'] = value
                elif key in ('description', 'desc'):
                    current['desc'] = value

        # Only add if we have non-empty source, target, and relation
        source = current.get('source', '').strip()
        target = current.get('target', '').strip()
        relation = current.get('relation', '').strip()

        if source and target and relation:
            relations.append(current)

    return {"relations": relations}


def parse_entity_resolution(text: str) -> dict[str, str]:
    """
    Parse natural language entity resolution output.

    Expected format:
        dragon_rest -> dragon_s_rest
        area_a1 -> A1
        the_beach -> rocky_shore

    Returns:
        Mapping dict {old_id: new_id}
    """
    mapping = {}

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Parse "old_id -> new_id" or "old_id: new_id"
        if '->' in line:
            old_id, new_id = line.split('->', 1)
        elif ':' in line:
            old_id, new_id = line.split(':', 1)
        else:
            continue

        old_id = old_id.strip()
        new_id = new_id.strip()

        if old_id and new_id and old_id != new_id:
            mapping[old_id] = new_id

    return mapping


def parse_verification(text: str) -> dict[str, Any]:
    """
    Parse natural language verification output.

    Expected format:
        Is Complete: No
        Missing Locations: B2, C5
        Issues:
        - Missing ship deck C9
        - Cave B6 not connected to parent

    Returns:
        {"is_complete": bool, "missing_locations": [...], "issues": [...]}
    """
    result = {
        "is_complete": False,
        "missing_locations": [],
        "issues": []
    }

    lines = text.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Check for section headers
        line_lower = line.lower()
        if 'is complete' in line_lower or 'complete:' in line_lower:
            result['is_complete'] = 'yes' in line_lower or 'true' in line_lower
            current_section = None
        elif 'missing' in line_lower and 'location' in line_lower:
            current_section = 'missing_locations'
        elif 'issue' in line_lower:
            current_section = 'issues'
        elif line.startswith('-'):
            # List item
            item = line.lstrip('-').strip()
            if current_section == 'missing_locations':
                result['missing_locations'].append(item)
            elif current_section == 'issues':
                result['issues'].append(item)

    return result


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Fallback: Try to extract JSON from mixed text+JSON output.

    Useful when LLM outputs some text followed by JSON.
    """
    # Look for JSON blocks between ```json and ``` or between { and }
    json_pattern = r'```json\s*(.*?)\s*```|(\{.*\})'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        json_str = match[0] if match[0] else match[1]
        try:
            import json
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue

    return None
