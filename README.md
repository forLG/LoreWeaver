# LoreWeaver

An automated pipeline to convert D&D module PDFs/Texts into Knowledge Graphs for GraphRAG, powering an AI Dungeon Master.

## Installation

See pyproject.toml for dependencies.

```bash
pip install -e .
```

Configure environment variables in `.env`:

```bash
# Required: LLM API configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Or your vLLM endpoint
LLM_MODEL=qwen3-8b  # Model name

# Optional: Concurrency control
LLM_MAX_CONCURRENT=5  # For local models, default: 5

# Optional: Neo4j (for graph storage/ querying)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Pipeline Overview

LoreWeaver supports two pipelines:

1. **Large Model Pipeline** (`main.py`) - For powerful models like GPT-4, Claude, DeepSeek-Chat
2. **Small Model Pipeline** (`main_semantic.py`) - Optimized for 7B-8B models like Qwen3-8B, Llama-3-8B

```
                    ┌─────────────────────┐
                    │   5eTools JSON      │
                    │   (Adventure Data)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Preprocessor      │
                    │  - Parse structure  │
                    │  - Extract tags     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌───────────────────┐           ┌─────────────────────┐
    │  Large Model     │           │   Small Model       │
    │  Pipeline        │           │   Pipeline          │
    │  (main.py)       │           │   (main_semantic.py)│
    └───────────────────┘           └─────────────────────┘
    │ Shadow Tree     │             │ Content Tree        │
    │                 │             │ Unified Extraction  │
    │ JSON output     │             │ Heterogeneous Graph │
    └───────────────────┘           └─────────┬───────────┘
                                             │
                                      ┌──────▼──────┐
                                      │  JSON Graph │
                                      │  + Events   │
                                      └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │  Neo4j/HTML │
                                      │  (Optional) │
                                      └─────────────┘
```

---

## Small Model Pipeline (Recommended for Qwen3-8B, Llama-3-8B)

The small model pipeline is designed for efficient processing with limited context windows and focuses on semantic entity extraction with event tracking.

### Architecture

The pipeline uses a **unified heterogeneous graph approach** that extracts entities and events together:

```
Phase 1: Unified Entity + Event Extraction
   ├── Preprocessor: Extract seed entities from {@creature}, {@item}, {@spell} tags
   ├── LLM: Extract entities and events from each content node
   └── Output: Raw entities + events with source tracking

Phase 2: Entity Resolution
   ├── Deduplicate entities by ID
   ├── Filter meaningless entities (location_1, entity_2, etc.)
   ├── Build entity ID mapping for fuzzy matching
   └── Output: Unique, meaningful entities

Phase 2.5: Location Hierarchy Extraction
   └── Extract part_of relationships between locations

Phase 2.6: Semantic Relation Extraction
   └── Extract domain-specific relations (inhabits, guards, hidden_at, etc.)

Phase 3: Event Processing
   ├── Update event entity references to canonical IDs
   ├── Generate edges from events to participants/locations
   └── Output: Valid events with connected edges

Phase 4: Graph Construction
   ├── Build validated heterogeneous graph
   ├── Pydantic validation (with fallback)
   └── Output: Final semantic graph JSON
```

### Key Features

- **Natural Language Output**: Uses XML-tagged output (`<entities>`, `<events>`) instead of JSON for robustness with small models
- **List-Style NPC Extraction**: Extracts individual characters from list-format descriptions
- **Generic Creature Handling**: Marks groups like "zombies", "guards" with `is_generic: true`
- **Event Extraction**: Captures encounters, combat, discovery, dialogue events
- **Seed Entities**: Uses preprocessor-extracted entities as ground-truth context for LLM
- **Robust Parsing**: Handles truncated responses and missing closing tags

### Usage

#### Basic Usage

```bash
# Run with default model (from .env or config)
python -m main_semantic

# Use specific model (local vLLM or API)
python -m main_semantic --model qwen3-8b --base-url http://localhost:8000/v1

# Rerun, ignore cache
python -m main_semantic --force

# Dry run (preview without processing)
python -m main_semantic --dry-run
```

#### Advanced Options

```bash
# Specify input/output
python -m main_semantic \
    --input data/adventure-dosi.json \
    --output-dir output/qwen3

# Adjust concurrency (important for local models)
python -m main_semantic --max-concurrent 10

# Adjust max tokens for LLM output
python -m main_semantic --max-tokens 4096

# Adjust repetition penalty (for local models)
python -m main_semantic --repetition-penalty 1.1
```

#### Recommended Settings

**For local vLLM deployment (Qwen3-8B):**
```bash
python -m main_semantic \
    --model ./qwen3-8b-local/Qwen/Qwen3-8B \
    --base-url http://localhost:8000/v1 \
    --max-concurrent 5 \
    --max-tokens 8192
```

**For API-based models (DeepSeek, OpenAI):**
```bash
python -m main_semantic \
    --model deepseek-chat \
    --base-url https://api.deepseek.com/v1 \
    --max-concurrent 50 \
    --max-tokens 4096
```

### Output Files

```
output/qwen3/
├── adventure_parsed.json           # Parsed adventure structure (internal_index)
├── adventure_resolved.json         # Resolved seed entities from tags
├── semantic_graph.json             # Final heterogeneous graph (nodes + edges)
└── semantic_debug/                 # Debug outputs for each phase
    ├── debug_phase1_unified_entities.json
    ├── debug_phase1_unified_events.json
    ├── debug_phase2_5_location_hierarchies.json
    └── debug_phase2_6_semantic_relations.json
```

### Semantic Graph Structure

The `semantic_graph.json` contains a **heterogeneous graph** with:

**Nodes:**
- **Entity Nodes** (type: "entity"): Creatures, Locations, Items, Groups
- **Event Nodes** (type: "event"): Encounters, combat, discovery, dialogue events

**Edges:**
- **Spatial**: `part_of`, `connected_to`, `leads_to`
- **State**: `inhabits`, `guards`, `hidden_at`, `stored_in`
- **Social**: `commands`, `serves`, `allied_with`, `rival_of`
- **Event Relations**: `has_participant`, `occurs_at`

### Evaluation

Evaluate extraction quality against human labels:

```bash
python -m scripts.evaluation.evaluate_extraction \
    --graph output/qwen3/semantic_graph.json \
    --labels tests/samples/Stormwreck_Isle.txt \
    --threshold 80
```

Debug extraction for a specific node:

```bash
python -m scripts.evaluation.debug_extraction \
    --node-id 01e \
    --parsed output/qwen3/adventure_parsed.json
```

Compare extracted entities side-by-side with human labels:

```bash
python -m scripts.evaluation.compare_entities \
    --graph output/qwen3/semantic_graph.json \
    --labels tests/samples/Stormwreck_Isle.txt
```

### Troubleshooting

**Issue: Low entity recall (missing NPCs)**
- Check if NPCs are described in list-style format
- Verify prompt includes "LIST-STYLE NPC DESCRIPTIONS" rule
- Run debug_extraction on specific node to see LLM output

**Issue: Events not being parsed**
- Ensure `stop=None` in unified extraction (no early stopping)
- Check logs for "Extracted N raw events" > 0

**Issue: Pydantic validation errors**
- Check for entities missing `label` field
- Fallback returns unvalidated graph (processing continues)

---

## Large Model Pipeline (GPT-4, Claude, DeepSeek-Chat)

The large model pipeline uses multi-pass extraction for spatial hierarchies and entity enrichment.

### Usage

```bash
# Run all stages
python -m main --stage all

# Run specific stages
python -m main --stage shadow    # Build shadow tree
python -m main --stage spatial   # Extract location hierarchies
python -m main --stage entity    # Extract entities and relations

# Rerun all, ignore cache
python -m main --stage all --force

# Dry run to preview
python -m main --stage all --dry-run

# Override config via CLI
python -m main --stage entity --model deepseek-chat
```

### Graph Building & Visualization

#### Data Flow

```
                    ┌─────────────────────────────────────────┐
                    │           JSON Graph Files              │
                    │  ┌──────────────────┐  ┌──────────────┐ │
                    │  │ semantic_graph   │  │    (other)   │ │
                    │  │      .json       │  │    graphs    │ │
                    │  └────────┬─────────┘  └──────┬───────┘ │
                    └─────────────┼─────────────────┼─────────┘
                                  │                 │
                 ┌────────────────┘                 └────────────────┐
                 │                                                   │
                 ▼                                                   ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │      Neo4j Database       │                   │     HTML Visualization    │
    │  (Persistent Graph Store) │                   │     (Interactive View)    │
    └───────────────────────────┘                   └───────────────────────────┘
```

#### Usage

##### 1. Visualize Semantic Graph (Small Model Pipeline)

For the heterogeneous graph from `main_semantic.py`:

```bash
# Full graph visualization
python scripts/visualize_semantic_graph.py --graph output/qwen3/semantic_graph.json

# Custom output path
python scripts/visualize_semantic_graph.py --graph output/qwen3/semantic_graph.json --output my_graph.html

# Only show statistics
python scripts/visualize_semantic_graph.py --graph output/qwen3/semantic_graph.json --stats-only

# Filtered visualizations (show only specific types)
python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter creature,event
python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter location
python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter item
```

Filter options: `creature`, `location`, `item`, `group`, `event` (comma-separated for multiple).

Output: Interactive HTML file with zoom, pan, and hover tooltips.

##### 2. Visualize Large Model Pipeline Graphs

For the separate location/entity graphs from `main.py`:

```bash
python scripts/build_graph.py --mode visualize
```

Output: `output/visualizations/{location,entity,combined}_graph.html`

##### 3. Import to Neo4j (Optional, for GraphRAG/Querying)

```bash
# Install and start Neo4j first
python scripts/build_graph.py --mode neo4j
# or do both visualization and Neo4j import
python scripts/build_graph.py --mode both
```

##### 4. Visualize from Neo4j with Custom Queries

```bash
# After importing to Neo4j, use Cypher queries to filter data
python scripts/visualize_from_neo4j.py
```

#### Configuration

Edit `config_neo4j.py`:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"  # Change this
<<<<<<< HEAD
``` -->

### frontend

To run the Streamlit frontend:

```
python -m streamlit run frontend/app.py
```
