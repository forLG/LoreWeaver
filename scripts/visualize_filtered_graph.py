"""
Generate filtered visualizations of the semantic graph.

Usage:
    # Show only creatures and their relations
    python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter creature

    # Show only locations
    python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter location

    # Show only events
    python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter event

    # Show creatures + events (NPC interactions)
    python scripts/visualize_filtered_graph.py --graph output/qwen3/semantic_graph.json --filter creature,event
"""
# ruff: noqa: E402 - Module level imports not at top (required for sys.path modification)
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.visualize_semantic_graph import generate_html_visualization, load_semantic_graph


def filter_graph(graph: dict, filters: list[str]) -> dict:
    """Filter the graph to only include specific node types."""
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    # Map filters to node criteria
    filter_map = {
        'creature': lambda n: n.get('node_type') == 'entity' and n.get('type') == 'Creature',
        'location': lambda n: n.get('node_type') == 'entity' and n.get('type') == 'Location',
        'item': lambda n: n.get('node_type') == 'entity' and n.get('type') == 'Item',
        'group': lambda n: n.get('node_type') == 'entity' and n.get('type') == 'Group',
        'event': lambda n: n.get('node_type') == 'event',
    }

    # Combine filters
    def should_include_node(node):
        for f in filters:
            f_lower = f.lower()
            if f_lower in filter_map and filter_map[f_lower](node):
                return True
        return False

    # Filter nodes
    filtered_nodes = [n for n in nodes if should_include_node(n)]
    node_ids = {n.get('id') for n in filtered_nodes}

    # Filter edges - keep edges where both ends are in the filtered nodes
    filtered_edges = [
        e for e in edges
        if e.get('source') in node_ids and e.get('target') in node_ids
    ]

    return {'nodes': filtered_nodes, 'edges': filtered_edges}


def main():
    parser = argparse.ArgumentParser(description='Generate filtered visualizations of semantic graph')
    parser.add_argument('--graph', default='output/qwen3/semantic_graph.json',
                        help='Path to semantic_graph.json')
    parser.add_argument('--filter', required=True,
                        help='Comma-separated filter: creature,location,item,group,event')
    parser.add_argument('--output', default='output/filtered_graph.html',
                        help='Output HTML file path')

    args = parser.parse_args()

    # Parse filters
    filters = [f.strip() for f in args.filter.split(',')]

    # Load and filter graph
    print(f"Loading graph from: {args.graph}")
    graph = load_semantic_graph(args.graph)

    print(f"Filtering by: {', '.join(filters)}")
    filtered_graph = filter_graph(graph, filters)

    print("Filtered graph:")
    print(f"  Nodes: {len(filtered_graph['nodes'])}")
    print(f"  Edges: {len(filtered_graph['edges'])}")

    # Generate visualization
    generate_html_visualization(filtered_graph, args.output)
    print(f"\nFiltered visualization saved to: {args.output}")


if __name__ == '__main__':
    main()
