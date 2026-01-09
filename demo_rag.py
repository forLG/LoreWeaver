import os
from src.graphRAG.graph_loader import GraphLoader
from src.graphRAG.rag_engine import SimpleGraphRAG

def main():
    """Main entry point for the Graph RAG demonstration."""
    # 1. Path configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'output', 'deepseek', 'entity_graph.json')
    
    # 2. Load graph data
    print(f"Loading graph from {data_path}...")
    loader = GraphLoader(data_path)
    try:
        graph = loader.load_graph()
        stats = loader.get_stats()
        print(f"Graph Loaded Successfully!")
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # 3. Initialize RAG Engine
    rag = SimpleGraphRAG(graph)

    # 4. Demo Query
    test_query = "where are zombies found?"
    print(f"\n--- Testing Query: '{test_query}' ---")
    
    # Get generated prompt (with context)
    prompt = rag.answer_query(test_query)
    
    print("Generated Prompt Context:")
    print("-" * 30)
    print(prompt)
    print("-" * 30)
    print("Next Step: Connect this prompt to an LLM (e.g., OpenAI, Ollama).")

if __name__ == "__main__":
    main()
