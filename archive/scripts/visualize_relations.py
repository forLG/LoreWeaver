#!/usr/bin/env python3
"""
Visualize knowledge graph relations from debug output.

Usage:
    python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json
    python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json --output graph.html
    python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json --filter "part_of|inhabits"
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install networkx pyvis")
    raise SystemExit(1)


def load_relations(filepath: str | Path) -> list[dict]:
    """Load relations from JSON file."""
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def filter_relations(relations: list[dict], pattern: str | None) -> list[dict]:
    """Filter relations by regex pattern on relation type."""
    if not pattern:
        return relations

    regex = re.compile(pattern)
    return [r for r in relations if regex.search(r.get('relation', ''))]


def count_relation_types(relations: list[dict]) -> dict[str, int]:
    """Count occurrences of each relation type."""
    counts: dict[str, int] = {}
    for r in relations:
        rel_type = r.get('relation', 'unknown')
        counts[rel_type] = counts.get(rel_type, 0) + 1
    return counts


def build_graph(relations: list[dict]) -> nx.DiGraph:
    """Build NetworkX graph from relations."""
    G = nx.DiGraph()

    # Track entity info for node attributes
    entity_info: dict[str, dict[str, Any]] = {}

    for rel in relations:
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('relation', 'related_to')
        desc = rel.get('desc', '')
        source_node = rel.get('source_node', '')

        # Skip unknown/invalid entities
        if not source or not target or source in ('<unknown>', 'null', 'undefined_target'):
            continue
        if target in ('<unknown>', 'null'):
            continue

        # Track entity sources
        if source not in entity_info:
            entity_info[source] = {'sources': set()}
        if source_node:
            entity_info[source]['sources'].add(source_node)

        if target not in entity_info:
            entity_info[target] = {'sources': set()}
        if source_node:
            entity_info[target]['sources'].add(source_node)

        # Add edge with attributes
        edge_key = (source, target)
        if G.has_edge(*edge_key):
            # Multiple relations between same nodes - combine them
            existing = G.edges[edge_key]
            existing_relations = existing.get('relations', [existing.get('relation', '')])
            existing_descriptions = existing.get('descriptions', [existing.get('description', '')])

            existing_relations.append(rel_type)
            existing_descriptions.append(desc)

            G.edges[edge_key]['relations'] = existing_relations
            G.edges[edge_key]['descriptions'] = existing_descriptions
            G.edges[edge_key]['description'] = f"{len(existing_relations)} relations"
        else:
            G.add_edge(
                source,
                target,
                relation=rel_type,
                description=desc,
                relations=[rel_type],
                descriptions=[desc],
                title=f"{rel_type}\n{desc}"  # Tooltip
            )

    # Add node attributes
    for node, info in entity_info.items():
        G.nodes[node]['title'] = node  # Tooltip
        G.nodes[node]['label'] = node

    return G


def get_node_color_by_degree(G: nx.DiGraph, node: str) -> str:
    """Color node based on degree (centrality)."""
    degree = G.degree(node)

    if degree >= 20:
        return '#ff6b6b'  # Red - highly connected
    elif degree >= 10:
        return '#feca57'  # Orange - well connected
    elif degree >= 5:
        return '#48dbfb'  # Blue - moderately connected
    else:
        return '#1dd1a1'  # Green - less connected


def create_pyvis_graph(
    G: nx.DiGraph,
    output_path: str | Path = "graph.html",
    height: str = "900px",
    width: str = "100%"
) -> None:
    """Create interactive Pyvis visualization."""
    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0

    # Create Pyvis network
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor='#1a1a2e',
        font_color='#ffffff',
        notebook=False
    )

    # Add nodes
    for node, data in G.nodes(data=True):
        degree = degrees[node]
        # Scale size: 10-50 based on degree
        normalized = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
        size = 10 + int(normalized * 40)

        net.add_node(
            node,
            label=data.get('label', node),
            title=data.get('title', node),
            size=size,
            color=get_node_color_by_degree(G, node)
        )

    # Add edges
    for source, target, data in G.edges(data=True):
        relations = data.get('relations', [data.get('relation', 'related_to')])
        descriptions = data.get('descriptions', [data.get('description', '')])

        # Build tooltip
        if len(relations) > 1:
            title = f"{len(relations)} relations:\n" + "\n".join(
                f"  {r}: {d[:50]}..." if len(d) > 50 else f"  {r}: {d}"
                for r, d in zip(relations, descriptions)
            )
        else:
            title = data.get('title', f"{relations[0]}\n{descriptions[0]}")

        net.add_edge(
            source,
            target,
            title=title,
            label=relations[0] if len(relations) == 1 else f"{len(relations)} relations",
            arrowStrikethrough=False
        )

    # Set physics options for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "timestep": 0.5,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      },
      "nodes": {
        "font": {
          "size": 12,
          "face": "Arial",
          "color": "#ffffff"
        },
        "borderWidth": 2,
        "borderWidthSelected": 3
      },
      "edges": {
        "width": 1,
        "smooth": {
          "type": "continuous"
        },
        "color": {
          "color": "#888888",
          "highlight": "#ff6b6b",
          "hover": "#feca57"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragView": true
      }
    }
    """)

    # Save and show
    output_file = str(output_path)
    net.save_graph(output_file)
    print(f"Graph saved to: {output_file}")


def print_statistics(relations: list[dict], G: nx.DiGraph) -> None:
    """Print graph statistics."""
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)

    # Relation statistics
    print(f"\nTotal relations: {len(relations)}")
    print(f"Unique nodes: {G.number_of_nodes()}")
    print(f"Unique edges: {G.number_of_edges()}")

    # Relation type counts
    print("\n--- Relation Types ---")
    counts = count_relation_types(relations)
    for rel_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type}: {count}")

    # Top connected nodes
    print("\n--- Top Connected Nodes ---")
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20]
    for node, degree in degrees:
        print(f"  {node}: {degree} connections")

    # Graph properties
    print("\n--- Graph Properties ---")
    if nx.is_weakly_connected(G):
        print("  Weakly connected: Yes")
        print(f"  Diameter: {nx.diameter(G.to_undirected())}")
    else:
        components = list(nx.weakly_connected_components(G))
        print(f"  Weakly connected: No ({len(components)} components)")
        print(f"  Largest component: {max(len(c) for c in components)} nodes")

    avg_clustering = nx.average_clustering(G.to_undirected())
    print(f"  Avg clustering: {avg_clustering:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize knowledge graph relations from debug output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json

  # Specify output file
  python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json -o mygraph.html

  # Filter by relation type (regex)
  python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json -f "part_of|inhabits"

  # Filter by relation type and save
  python visualize_relations.py output/qwen3/debug_phase1_combined_relations.json -f "attacks" -o combat_graph.html
        """
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to relations JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="graph.html",
        help="Output HTML file (default: graph.html)"
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        help="Filter relations by regex pattern (e.g., 'part_of|inhabits')"
    )
    parser.add_argument(
        "--height",
        type=str,
        default="900px",
        help="Graph height (default: 900px)"
    )
    parser.add_argument(
        "--width",
        type=str,
        default="100%",
        help="Graph width (default: 100%%)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't print statistics"
    )

    args = parser.parse_args()

    # Load relations
    print(f"Loading relations from: {args.input}")
    relations = load_relations(args.input)

    # Filter if requested
    if args.filter:
        print(f"Filtering relations by pattern: {args.filter}")
        relations = filter_relations(relations, args.filter)
        print(f"Filtered to {len(relations)} relations")

    # Build graph
    print("Building graph...")
    G = build_graph(relations)

    # Print statistics
    if not args.no_stats:
        print_statistics(relations, G)

    # Create visualization
    print("\nCreating visualization...")
    create_pyvis_graph(G, args.output, args.height, args.width)

    print(f"\nDone! Open {args.output} in your browser to view the graph.")


if __name__ == "__main__":
    main()
