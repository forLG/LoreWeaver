"""
Visualize the semantic graph from the small model pipeline.

Generates an interactive HTML visualization using pyvis.

Usage:
    python scripts/visualize_semantic_graph.py --graph output/qwen3/semantic_graph.json
    python scripts/visualize_semantic_graph.py --graph output/qwen3/semantic_graph.json --output output/visualization.html
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_semantic_graph(graph_path: str) -> dict:
    """Load the semantic graph JSON file."""
    with open(graph_path, encoding='utf-8') as f:
        return json.load(f)


def print_graph_stats(graph: dict) -> None:
    """Print statistics about the semantic graph."""
    print("=" * 60)
    print("SEMANTIC GRAPH STATISTICS")
    print("=" * 60)

    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    # Count by node type
    entity_nodes = [n for n in nodes if n.get('node_type') == 'entity']
    event_nodes = [n for n in nodes if n.get('node_type') == 'event']

    print(f"Total nodes: {len(nodes)}")
    print(f"  Entity nodes: {len(entity_nodes)}")
    print(f"  Event nodes: {len(event_nodes)}")
    print(f"Total edges: {len(edges)}")

    # Entity type distribution
    if entity_nodes:
        print("\nEntity type distribution:")
        entity_types = Counter(n.get('type', 'Unknown') for n in entity_nodes)
        for etype, count in entity_types.most_common():
            print(f"  {etype}: {count}")

    # Event type distribution
    if event_nodes:
        print("\nEvent type distribution:")
        event_types = Counter(n.get('type', 'Unknown') for n in event_nodes)
        for etype, count in event_types.most_common():
            print(f"  {etype}: {count}")

    # Edge relation distribution
    if edges:
        print("\nEdge relation types:")
        relations = Counter(e.get('relation', 'unknown') for e in edges)
        for rel, count in relations.most_common(20):
            print(f"  {rel}: {count}")


def generate_html_visualization(graph: dict, output_path: str) -> None:
    """Generate an HTML visualization using a simple template."""
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    # Create node data for vis.js
    vis_nodes = []
    for node in nodes:
        node_id = node.get('id')
        # Handle missing or None labels
        label = node.get('label') or node.get('id', 'Unknown')
        if label is None or label == 'None':
            label = node.get('id', 'Unknown')
        node_type = node.get('node_type', 'entity')
        entity_type = node.get('type', 'Unknown')

        # Choose color based on type
        if node_type == 'event':
            color = '#FF6B6B'  # Red for events
        else:
            # Color by entity type
            color_map = {
                'Creature': '#4ECDC4',
                'Location': '#45B7D1',
                'Item': '#FFA07A',
                'Group': '#98D8C8',
            }
            color = color_map.get(entity_type, '#CCCCCC')

        vis_nodes.append({
            'id': node_id,
            'label': label,
            'title': f"{label}\nType: {entity_type}\nID: {node_id}",
            'color': color,
            'font': {'size': 14},
        })

    # Create edge data
    vis_edges = []
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        relation = edge.get('relation', 'unknown')
        desc = edge.get('desc', '')

        vis_edges.append({
            'from': source,
            'to': target,
            'label': relation,
            'title': f"{relation}\n{desc}",
            'font': {'size': 10},
            'arrows': 'to',
        })

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>LoreWeaver Semantic Graph Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1 {{
            color: #333;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin: 5px 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <h1>LoreWeaver Semantic Graph</h1>

    <div class="legend">
        <strong>Legend:</strong><br>
        <span class="legend-item"><span class="legend-color" style="background: #4ECDC4;"></span>Creature</span>
        <span class="legend-item"><span class="legend-color" style="background: #45B7D1;"></span>Location</span>
        <span class="legend-item"><span class="legend-color" style="background: #FFA07A;"></span>Item</span>
        <span class="legend-item"><span class="legend-color" style="background: #98D8C8;"></span>Group</span>
        <span class="legend-item"><span class="legend-color" style="background: #FF6B6B;"></span>Event</span>
    </div>

    <div id="mynetwork"></div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(vis_nodes, indent=8)});
        var edges = new vis.DataSet({json.dumps(vis_edges, indent=8)});

        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 16,
                font: {{
                    size: 14,
                    color: '#333333'
                }}
            }},
            edges: {{
                width: 1,
                smooth: {{ type: 'continuous' }},
                font: {{
                    size: 10,
                    align: 'middle'
                }}
            }},
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true
            }}
        }};

        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nVisualization saved to: {output_path}")
    print("Open the file in a web browser to explore the graph.")


def main():
    parser = argparse.ArgumentParser(description='Visualize semantic graph from small model pipeline')
    parser.add_argument('--graph', default='output/qwen3/semantic_graph.json',
                        help='Path to semantic_graph.json')
    parser.add_argument('--output', default='output/semantic_graph_visualization.html',
                        help='Output HTML file path')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, don\'t generate visualization')

    args = parser.parse_args()

    # Load graph
    print(f"Loading graph from: {args.graph}")
    graph = load_semantic_graph(args.graph)

    # Print stats
    print_graph_stats(graph)

    # Generate visualization
    if not args.stats_only:
        generate_html_visualization(graph, args.output)


if __name__ == '__main__':
    main()
