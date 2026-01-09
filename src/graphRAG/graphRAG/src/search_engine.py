import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class GraphSearchEngine:
    """Handles graph search operations including vector search and traversal.

    Attributes:
        graph (nx.Graph): The knowledge graph.
        model (SentenceTransformer): Embedding model.
        node_embeddings (np.array): Precomputed embeddings for nodes.
        node_ids (list): List of node IDs corresponding to embeddings.
        edge_embeddings (np.array): Precomputed embeddings for edges.
        edge_data (list): List of edge tuples (source, target, relation, desc).
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._init_embedding_model()
        self._precompute_embeddings()
        self._precompute_edge_embeddings()

    def _init_embedding_model(self):
        """Initializes the embedding model from local or remote source."""
        model_name = 'all-MiniLM-L6-v2'
        # Go up two levels from src/ to root, then into models/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(base_dir, 'models', model_name)

        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"Loading local embedding model: {local_path}")
            self.model = SentenceTransformer(local_path)
        else:
            print(f"Loading/Downloading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)

    def _precompute_embeddings(self):
        """Computes embeddings for all graph nodes."""
        self.node_ids = []
        texts = []
        for node, data in self.graph.nodes(data=True):
            self.node_ids.append(node)
            label = data.get("label", "")
            node_type = data.get("type", "")
            texts.append(f"{label} ({node_type})")
        
        if texts:
            self.node_embeddings = self.model.encode(texts)
        else:
            self.node_embeddings = np.array([])

    def vector_search(self, query: str, top_k=3):
        """Finds top_k nodes semantically similar to the query."""
        if len(self.node_embeddings) == 0:
            return []

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.node_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.node_ids[i] for i in top_indices]

    def get_subgraph_context(self, start_nodes, max_hops=1):
        """Traverses the graph from start_nodes (BFS/Ego Graph)."""
        context_lines = []
        visited = set()

        for node in start_nodes:
            subgraph = nx.ego_graph(self.graph, node, radius=max_hops)
            for u, v, data in subgraph.edges(data=True):
                if (u, v) in visited: 
                    continue
                visited.add((u, v))
                
                s_lbl = self.graph.nodes[u].get('label', u)
                t_lbl = self.graph.nodes[v].get('label', v)
                rel = data.get('relationship', 'related to')
                desc = data.get('desc', '')
                
                line = f"- {s_lbl} --[{rel}]--> {t_lbl}"
                if desc: line += f" ({desc})"
                context_lines.append(line)
        
        return context_lines

    def get_shortest_paths(self, nodes):
        """Finds shortest paths between pairs of nodes in the list."""
        paths_context = []
        if len(nodes) < 2:
            return paths_context

        paths_context.append("\n--- Connections between entities ---")
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                source, target = nodes[i], nodes[j]
                try:
                    path = nx.shortest_path(self.graph, source, target)
                    if len(path) > 4: continue # Skip long paths

                    # Format path
                    path_desc = []
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k+1]
                        edge_data = self.graph.get_edge_data(u, v)
                        rel = 'connected'
                        if edge_data:
                            # Handle DiGraph vs MultiGraph if needed
                            rel = edge_data.get('relationship', 'connected')
                        
                        u_lbl = self.graph.nodes[u].get('label', u)
                        v_lbl = self.graph.nodes[v].get('label', v)
                        path_desc.append(f"{u_lbl} --[{rel}]--> {v_lbl}")
                    
                    paths_context.append(f"Connection: {' '.join(path_desc)}")
                except nx.NetworkXNoPath:
                    continue
        return paths_context

    def _precompute_edge_embeddings(self):
        """Computes embeddings for all graph edges using description text."""
        self.edge_data = []
        texts = []

        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relationship', '')
            desc = data.get('desc', '')

            # Store edge data for later retrieval
            self.edge_data.append((u, v, relation, desc))

            # Use description as the text representation for embedding
            texts.append(desc)

        if texts:
            self.edge_embeddings = self.model.encode(texts)
        else:
            self.edge_embeddings = np.array([])

    def edge_vector_search(self, query: str, top_k=3):
        """Finds top_k edges semantically similar to the query.

        Args:
            query (str): The query text.
            top_k (int): Number of edges to return.

        Returns:
            list: List of edge tuples (source, target, relation, desc).
        """
        if len(self.edge_embeddings) == 0:
            return []

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.edge_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.edge_data[i] for i in top_indices]

    def get_edge_context(self, matched_edges):
        """Formats matched edges into readable context strings.

        Args:
            matched_edges (list): List of edge tuples (source, target, relation, desc).

        Returns:
            list: Formatted context strings.
        """
        context_lines = []

        for u, v, relation, desc in matched_edges:
            s_lbl = self.graph.nodes[u].get('label', u)
            t_lbl = self.graph.nodes[v].get('label', v)

            line = f"- {s_lbl} --[{relation}]--> {t_lbl}"
            if desc:
                line += f" ({desc})"
            context_lines.append(line)

        return context_lines
