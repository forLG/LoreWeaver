"""
Natural Language Parsers for LLM outputs.

Simplified to extract only core node attributes:
- id: Unique identifier
- label: Display name
- type: Entity type (location, creature, item, event, etc.)
- aliases: Alternative names

All other attributes (is_generic, location_type, creature_type, etc.) are removed.
"""
import re
from typing import Any

from src.utils.logger import logger


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
        Entity: Zombies in Ship
        Type: Creature
        ID: zombies_in_ship
        Aliases: zombies
        </entities>

    Or if no entities found:
        <summary>
        Text contains no extractable entities.
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
            aliases = current.get('aliases', [])

            if label or eid:
                # If no label, try to use first alias as label
                if not label and aliases:
                    label = str(aliases[0]).strip('"\'')
                    current['label'] = label
                # If still no label, use formatted ID as label
                if not label and eid:
                    logger.warning(f"Entity '{eid}' has no label, using ID as label")
                    label = eid.replace('_', ' ').title()
                    current['label'] = label
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
                # Strip brackets if present: [zombies, undead] -> zombies, undead
                value = value.strip()
                if value.startswith('['):
                    value = value[1:]
                if value.endswith(']'):
                    value = value[:-1]
                # Split by comma, strip whitespace
                aliases = [a.strip() for a in value.split(',') if a.strip()]
                current['aliases'] = aliases

    # Don't forget the last entity
    label = current.get('label', '').strip()
    eid = current.get('id', '').strip()
    aliases = current.get('aliases', [])
    if label or eid:
        if not label and aliases:
            label = str(aliases[0]).strip('"\'')
            current['label'] = label
        if not label and eid:
            logger.warning(f"Entity '{eid}' has no label, using ID as label")
            label = eid.replace('_', ' ').title()
            current['label'] = label
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
        # Fallback: try to extract events section without closing tag
        events_match = re.search(r'<events>(.*)$', text, re.DOTALL)

    # If still no match, assume the text is already the events content
    if not events_match:
        if re.search(r'^\s*Event:', text, re.MULTILINE):
            events_text = text
        else:
            return {"events": events}
    else:
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

        source = current.get('source', '').strip()
        target = current.get('target', '').strip()
        relation = current.get('relation', '').strip()

        if source and target and relation:
            relations.append(current)

    return {"relations": relations}


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
    events_match = re.search(r'<events>(.*?)</events>', text, re.DOTALL)
    if not events_match:
        events_match = re.search(r'<events>(.*)$', text, re.DOTALL)

    if events_match:
        events_text = events_match.group(1)
        events_result = parse_events(events_text)
        result["events"] = events_result.get("events", [])

    return result
