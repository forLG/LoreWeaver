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

    Expected format (compact, no blank lines):
        <entities>
        Entity: Dragon's Rest
        Type: Location
        ID: dragon_s_rest
        Aliases: temple, monastery
        Entity: Runara
        Type: Creature
        ID: runara
        Aliases: bronze dragon, elder
        </entities>

    Or if no entities found:
        <summary>
        Text contains no extractable entities. Only generic background descriptions.
        </summary>

    Returns:
        {"entities": [...], "summary": str or None}
    """
    # Check for summary section first (no entities found)
    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        return {"entities": [], "summary": summary}

    # Extract entities section if present
    entities_match = re.search(r'<entities>(.*?)</entities>', text, re.DOTALL)
    if entities_match:
        text = entities_match.group(1)

    entities = []
    current = {}

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this is a new entity block
        if line.lower().startswith('entity:'):
            # Save previous entity if valid
            label = current.get('label', '').strip()
            eid = current.get('id', '').strip()

            if label or eid:
                if not eid:
                    eid = label.lower().replace(' ', '_').replace("'", "")
                    current['id'] = eid
                if 'type' not in current:
                    current['type'] = 'Entity'
                if 'aliases' not in current:
                    current['aliases'] = []
                entities.append(current)

            # Start new entity
            current = {}

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

    # Don't forget the last entity
    label = current.get('label', '').strip()
    eid = current.get('id', '').strip()
    if label or eid:
        if not eid:
            eid = label.lower().replace(' ', '_').replace("'", "")
            current['id'] = eid
        if 'type' not in current:
            current['type'] = 'Entity'
        if 'aliases' not in current:
            current['aliases'] = []
        entities.append(current)

    return {"entities": entities, "summary": None}


def parse_events(text: str) -> dict[str, Any]:
    """
    Parse event extraction output.

    Expected format:
        <events>
        Event: Meeting Runara
        Type: encounter
        Participants: [adventurers, runara]
        Location: dragon_s_rest
        Description: The party meets a bronze dragon in the temple

        Event: Finding the Key
        Type: discovery
        Participants: [adventurers]
        Location: dragon_s_rest
        Description: While searching, the party discovers a rusty key
        </events>

    Returns:
        {"events": [...]}
    """
    events = []

    # Extract events section if present
    events_match = re.search(r'<events>(.*?)</events>', text, re.DOTALL)
    if not events_match:
        return {"events": events}

    events_text = events_match.group(1)
    current = {}

    for line in events_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this is a new event block
        if line.lower().startswith('event:'):
            # Save previous event if valid
            if current.get('label'):
                # Generate ID from label if not present
                if 'id' not in current:
                    label = current.get('label', '').lower()
                    current['id'] = re.sub(r'[^a-z0-9]+', '_', label).strip('_')

                # Set defaults
                if 'type' not in current:
                    current['type'] = 'event'
                if 'participants' not in current:
                    current['participants'] = []
                if 'location' not in current:
                    current['location'] = None
                if 'description' not in current:
                    current['description'] = ''

                events.append(current)

            # Start new event
            current = {}

        # Parse key: value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'event':
                current['label'] = value
            elif key == 'type':
                current['type'] = value
            elif key == 'id':
                current['id'] = value
            elif key == 'participants':
                # Parse list format: [item1, item2] or comma-separated
                value = value.strip('[]()')
                if value:
                    participants = [p.strip() for p in value.split(',')]
                    current['participants'] = participants
                else:
                    current['participants'] = []
            elif key == 'location':
                current['location'] = value if value else None
            elif key == 'description':
                current['description'] = value

    # Don't forget the last event
    if current.get('label'):
        if 'id' not in current:
            label = current.get('label', '').lower()
            current['id'] = re.sub(r'[^a-z0-9]+', '_', label).strip('_')
        if 'type' not in current:
            current['type'] = 'event'
        if 'participants' not in current:
            current['participants'] = []
        if 'location' not in current:
            current['location'] = None
        if 'description' not in current:
            current['description'] = ''
        events.append(current)

    return {"events": events}


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


def parse_combined_extraction(text: str) -> dict[str, Any]:
    """
    Parse combined entity and relation extraction output.

    Expected format:
        <entities>
        Entity: Dragon's Rest
        Type: Location
        ID: dragon_s_rest
        Aliases: [temple, monastery]

        Entity: Runara
        Type: Creature
        ID: runara
        Aliases: [bronze dragon]
        </entities>

        <relations>
        Relation: inhabits
        Source: runara
        Target: dragon_s_rest
        Description: Lives in the temple

        Relation: commands
        Source: goblin_boss
        Target: goblin_minion
        Description: Leads the goblins
        </relations>

    Or if no entities:
        <summary>
        Text contains no extractable entities.
        </summary>

    Returns:
        {"entities": [...], "relations": [...], "summary": str or None}
    """
    result = {"entities": [], "relations": [], "summary": None}

    # Check for summary first (no entities found)
    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()
        return result

    # Extract entities section
    entities_match = re.search(r'<entities>(.*?)</entities>', text, re.DOTALL)
    if entities_match:
        entities_text = entities_match.group(1)
        entities_result = parse_ner_entities(entities_text)
        result["entities"] = entities_result.get("entities", [])
        # Also capture summary if present in entities section
        result["summary"] = entities_result.get("summary")

    # Extract relations section
    relations_match = re.search(r'<relations>(.*?)</relations>', text, re.DOTALL)
    if relations_match:
        relations_text = relations_match.group(1)
        relations_result = parse_relations(relations_text)
        result["relations"] = relations_result.get("relations", [])

    return result


def parse_unified_extraction(text: str) -> dict[str, Any]:
    """
    Parse unified entity + event extraction output for heterogeneous graph.

    Expected format:
        <entities>
        Entity: Dragon's Rest
        Type: Location
        ID: dragon_s_rest
        Aliases: [temple]
        </entities>

        <events>
        Event: Meeting Runara
        Type: encounter
        Participants: [adventurers, runara]
        Location: dragon_s_rest
        Description: The party meets a bronze dragon
        </events>

    Or if no content:
        <summary>
        Text contains only generic atmospheric description.
        </summary>

    Returns:
        {"entities": [...], "events": [...], "summary": str or None}
    """
    result = {"entities": [], "events": [], "summary": None}

    # Check for summary first (no entities/events found)
    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()
        return result

    # Extract entities section
    entities_match = re.search(r'<entities>(.*?)</entities>', text, re.DOTALL)
    if entities_match:
        entities_text = entities_match.group(1)
        entities_result = parse_ner_entities(entities_text)
        result["entities"] = entities_result.get("entities", [])

    # Extract events section
    events_result = parse_events(text)
    result["events"] = events_result.get("events", [])

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
