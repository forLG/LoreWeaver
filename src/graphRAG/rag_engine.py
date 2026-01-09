import networkx as nx
from .search_engine import GraphSearchEngine
from .llm_service import LLMService

class SimpleGraphRAG:
    """Orchestrates the RAG pipeline: Retrieval -> Augmented -> Generation.

    Attributes:
        search_engine (GraphSearchEngine): Component for retrieving graph context.
        llm_service (LLMService): Component for generating answers.
    """

    def __init__(self, graph: nx.Graph):
        """Initializes the RAG engine.

        Args:
            graph (nx.Graph): The networkx graph object.
            api_key (str, optional): API key for LLM service.
            base_url (str, optional): Base URL for LLM API.
            model (str, optional): Model identifier. Defaults to "deepseek-chat".
            enable_llm (bool, optional): Whether to enable LLM service. Defaults to True.
        """
        print("Initializing Search Engine...")
        self.search_engine = GraphSearchEngine(graph)

        self.llm_service = None
        
        print("Initializing LLM Service...")
        self.llm_service = LLMService()
        

    def retrieve_context(self, query: str, max_hops=1, top_k=3, top_k_edges=2):
        """Retrieves and formats context from the graph using the search engine.

        Args:
            query (str): The user's query.
            max_hops (int): Radius for neighbors.
            top_k (int): Number of entry points to find.
            top_k_edges (int): Number of edges to retrieve.

        Returns:
            str: Formatted context string.
        """
        # 1. Vector Search for Nodes
        entry_nodes = self.search_engine.vector_search(query, top_k=top_k)
        print(f"Debug: Found entry nodes: {entry_nodes}")

        # 2. Vector Search for Edges
        matched_edges = self.search_engine.edge_vector_search(query, top_k=top_k_edges)
        print(f"Debug: Found matched edges: {len(matched_edges)}")

        if not entry_nodes and not matched_edges:
            return "No relevant context found in the graph."

        # 3. Graph Traversal (Context from Nodes)
        context = []
        if entry_nodes:
            node_context = self.search_engine.get_subgraph_context(entry_nodes, max_hops)
            context.extend(node_context)

        # 4. Format Edges (Context from Edges)
        if matched_edges:
            edge_context = self.search_engine.get_edge_context(matched_edges)
            context.extend(edge_context)

        # 5. Path Finding (Reasoning)
        # paths = self.search_engine.get_shortest_paths(entry_nodes)

        # full_context = context + paths
        full_context = context
        return "\n".join(full_context)

    def answer_query(self, query: str, use_llm: bool = True):
        """Answers the user query using retrieved context and optionally LLM.

        Args:
            query (str): The user's query.
            use_llm (bool, optional): Whether to use LLM for generation. Defaults to True.

        Returns:
            str: The generated answer or the prompt if LLM is unavailable.
        """
        context = self.retrieve_context(query)

        if use_llm and self.llm_service:
            return self.llm_service.generate_answer(query, context)
        else:
            # Fallback to prompt-only mode
            prompt = f"""Based on the following knowledge graph context, answer the user query.

Context:
{context}

User Query: {query}
"""
            return prompt
