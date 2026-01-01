from preprocessor.explore import explore

explorer = explore('data/adventure-dosi-parsed.json')

# === OVERVIEW ===
explorer.show_summary()              # Overall statistics

# === BY ENTITY TYPE ===
explorer.show_entities('creature')   # All unique creatures
explorer.show_entities('item')       # All unique items
explorer.show_entities('spell')      # All unique spells

# === BY SECTION ===
explorer.show_section('02c')         # Section metadata + entities
explorer.show_section('02c', show_text=True)  # Include text content

# === ENTITY LOCATIONS ===
explorer.show_entity_locations('Runara')  # Where does X appear?

# === HIERARCHY ===
explorer.show_hierarchy('021')       # Show location tree
explorer.show_roots()                # Show top-level sections