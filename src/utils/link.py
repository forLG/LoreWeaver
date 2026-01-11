import re
import json

class LinkProcessor:
    """Processor for parsing and cleaning text containing special link tags."""

    # Formatting tags that should not be considered as navigational links.
    FORMATTING_TAGS = {
        'b', 'i', 'u', 's', 'c', 'color', 'note', 'bold', 'italic', 
        'strikethrough', 'underline', 'comic', 'book', 'filter'
    }

    @staticmethod
    def parse_and_clean(text: str):
        """Parses the text to remove tags and extract links.
        
        1. Removes tag syntax, preserving display text for reading.
        2. Extracts meaningful navigational links (links) and removes duplicates.

        Args:
            text: The raw text string containing tags.

        Returns:
            A tuple containing:
                - cleaned_text (str): The cleaned text with tags removed.
                - links (list): A list of extracted link objects (dictionaries).
        """
        if not isinstance(text, str):
            return "", []

        extracted_links = []

        # Helper function to handle individual regex matches
        def replace_tag(match):
            full_tag = match.group(0)  # e.g., {@area map 2|021|x}
            tag_type = match.group(1)  # e.g., area
            content = match.group(2)   # e.g., map 2|021|x

            # Handle pipe-separated attributes
            parts = content.split('|')
            display_text = parts[0]    # First part is always the display text
            attributes = parts[1:] if len(parts) > 1 else []

            # If not a pure formatting tag, record it as a link
            if tag_type not in LinkProcessor.FORMATTING_TAGS:
                extracted_links.append({
                    "text": display_text,
                    "tag": tag_type,
                    "attrs": attributes
                })

            return display_text

        # Regex: matches {@tag content}
        # [^{}]+ ensures matching the innermost tags to handle nesting.
        pattern = re.compile(r'{@(\w+)\s+([^{}]+)}')

        current_text = text
        # Loop until no tags remain (handling nested tags like {@note {@b bold} text})
        while True:
            new_text, count = pattern.subn(replace_tag, current_text)
            if count == 0:
                break
            current_text = new_text

        # --- Deduplication logic ---
        # Use JSON serialization as key to deduplicate the list of dictionaries
        unique_links = []
        seen = set()
        for link in extracted_links:
            # Convert dict to immutable JSON string for set storage
            # sort_keys=True ensures dicts with same content but different order are treated as identical
            link_signature = json.dumps(link, sort_keys=True)
            if link_signature not in seen:
                seen.add(link_signature)
                unique_links.append(link)

        return current_text, unique_links