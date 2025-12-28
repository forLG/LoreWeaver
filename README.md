# LoreWeaver

An automated pipeline to convert D&D module PDFs/Texts into Knowledge Graphs for GraphRAG, powering an AI Dungeon Master.

## Pipeline Overview

```
5eTools JSON → Shadow Tree → Spatial/Entity Processing → JSON Graphs → Neo4j/HTML
```

## Graph Building & Visualization

### Data Flow

```
                    ┌─────────────────────────────────────────┐
                    │           JSON Graph Files              │
                    │  ┌──────────────────┐  ┌──────────────┐ │
                    │  │ location_graph   │  │ entity_graph │ │
                    │  │      .json       │  │    .json     │ │
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

### Usage

#### 1. Generate Visualizations (No Neo4j required)

```bash
python scripts/build_graph.py --mode visualize
```

Output: `output/visualizations/{location,entity,combined}_graph.html`

#### 2. Import to Neo4j (Optional, for GraphRAG/Querying)

```bash
# Install and start Neo4j first (see docs/NEO4J_SETUP.md)
# import graph into Neo4j format
python scripts/build_graph.py --mode neo4j
# or do both visualization and Neo4j import
python scripts/build_graph.py --mode both
```

#### 3. Visualize from Neo4j with Custom Queries

```bash
# After importing to Neo4j, use Cypher queries to filter data
python scripts/visualize_from_neo4j.py
```

### Configuration

Edit `config_neo4j.py`:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"  # Change this
```
