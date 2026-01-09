import json
import networkx as nx
import os

class GraphLoader:
    """Loads graph data from a JSON file into a NetworkX graph.

    Attributes:
        file_path (str): Path to the JSON file containing graph data.
        graph (nx.DiGraph): The directed graph structure.
    """
    def __init__(self, file_path):
        """Initializes the GraphLoader with a file path.

        Args:
            file_path (str): The absolute or relative path to the JSON data file.
        """
        self.file_path = file_path
        self.graph = nx.DiGraph()

    def load_graph(self):
        """Loads JSON data and constructs a NetworkX DiGraph.

        Raises:
            FileNotFoundError: If the specified file_path does not exist.

        Returns:
            nx.DiGraph: The constructed directed graph with nodes and edges.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add nodes
        for node in data.get("nodes", []):
            self.graph.add_node(
                node["id"],
                label=node.get("label", ""),
                type=node.get("type", "unknown")
            )

        # Add edges
        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                relationship=edge.get("relation", ""),
                desc=edge.get("desc", "")
            )
        
        return self.graph

    def get_stats(self):
        """Returns basic statistics of the graph.

        Returns:
            dict: A dictionary containing number of nodes, number of edges, and density.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph)
        }
