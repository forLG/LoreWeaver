# Simple Graph RAG Project

This project implements a basic Graph Retrieval-Augmented Generation (RAG) system using `location_graph.json`.

## Project Structure

```
graphRAG/
├── data/
│   └── location_graph.json  # Your data file
├── src/
│   ├── graph_loader.py      # Loads JSON to NetworkX graph
│   └── rag_engine.py        # Logic to retrieve context from the graph
├── main.py                  # Entry point script
└── requirements.txt         # Dependencies
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `sentence-transformers`.*

2. Run the specific main script:

   ```bash
   python main.py
   ```
   *Note: The first run will download the model.*

## How it works

1. **Graph Loading**: The `GraphLoader` reads the JSON file and converts it into a `networkx` directed graph.
2. **Vector Indexing**: Upon initialization, `SimpleGraphRAG` uses `sentence-transformers` to compute embeddings for all nodes.
3. **Hybrid Retrieval**:
   - **Vector Search**: The user query is embedded, and the top-k most similar nodes are found.
   - **Graph Traversal**: From these "entry nodes", the system traverses the graph (1-hop) to gather context.
4. **Prompt Generation**: It constructs a text prompt containing the retrieved graph knowledge.

## Next Steps

- Integrate with an actual LLM (like OpenAI or HuggingFace) in `rag_engine.py` to generate natural language answers.
- Implement vector embeddings for more advanced fuzzy node matching.
