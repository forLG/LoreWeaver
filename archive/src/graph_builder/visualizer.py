"""
Graph Visualizer for LoreWeaver

支持将 Neo4j 图谱导出为交互式 HTML 可视化。
"""
import json
from pathlib import Path
from typing import ClassVar

from pyvis.network import Network

from utils.logger import logger


class GraphVisualizer:
    """
    图谱可视化器

    支持：
    - 从 JSON 文件直接可视化（无需 Neo4j）
    - 从 Neo4j 读取数据可视化
    - 导出为独立的 HTML 文件
    - 配置节点颜色、大小、布局
    """

    # 默认颜色配置（按节点类型）
    DEFAULT_COLORS: ClassVar[dict[str, str]] = {
        'Location': '#3498db',      # 蓝色
        'Creature': '#e74c3c',      # 红色
        'Item': '#f39c12',          # 橙色
        'Spell': '#9b59b6',         # 紫色
        'Party': '#2ecc71',         # 绿色
        'Entity': '#95a5a6',        # 灰色
        'World': '#1abc9c',
        'Region': '#16a085',
        'City': '#34495e',
        'default': '#bdc3c7'
    }

    def __init__(
        self,
        width: str = "100%",
        height: str = "800px",
        bgcolor: str = "#222222",
        font_color: str = "white",
        physics: bool = True
    ):
        """
        初始化可视化器

        Args:
            width: 画布宽度
            height: 画布高度
            bgcolor: 背景颜色
            font_color: 字体颜色
            physics: 是否启用物理布局引擎
        """
        self.width = width
        self.height = height
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.physics = physics

    # ---------------------------------------------------------------------
    # 从 JSON 文件可视化
    # ---------------------------------------------------------------------

    def visualize_from_json(
        self,
        graph_file: str,
        output_html: str = "output/graph_visualization.html",
        node_filter: list[str] | None = None,
        max_nodes: int | None = None,
        title: str = "LoreWeaver Knowledge Graph"
    ) -> str:
        """
        从 JSON 图谱文件生成可视化

        Args:
            graph_file: location_graph.json 或 entity_graph.json
            output_html: 输出 HTML 文件路径
            node_filter: 只显示指定类型的节点（如 ['Location', 'Creature']）
            max_nodes: 最大节点数（用于子集采样）
            title: 图谱标题

        Returns:
            输出 HTML 文件的绝对路径
        """
        logger.info(f"读取图谱文件: {graph_file}")

        with open(graph_file, encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        logger.info(f"原始数据: {len(nodes)} 节点, {len(edges)} 边")

        # 过滤节点
        if node_filter:
            filtered_types = set(node_filter)
            node_ids = {n['id'] for n in nodes if n.get('type', '').title() in filtered_types}
            nodes = [n for n in nodes if n['id'] in node_ids]
            edges = [e for e in edges if e['source'] in node_ids and e['target'] in node_ids]
            logger.info(f"过滤后: {len(nodes)} 节点, {len(edges)} 边")

        # 采样节点
        if max_nodes and len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            node_ids = {n['id'] for n in nodes}
            edges = [e for e in edges if e['source'] in node_ids and e['target'] in node_ids]
            logger.info(f"采样后: {len(nodes)} 节点, {len(edges)} 边")

        # 创建 Pyvis 网络
        net = self._create_pyvis_network(title)

        # 添加节点
        for node in nodes:
            node_type = node.get('type', 'unknown').title()
            color = self.DEFAULT_COLORS.get(node_type, self.DEFAULT_COLORS['default'])
            label = node.get('label', node['id'])

            # 根据度数调整节点大小
            degree = sum(1 for e in edges if e['source'] == node['id'] or e['target'] == node['id'])
            size = 10 + min(degree * 2, 30)  # 基础大小 10，最大 40

            net.add_node(
                node['id'],
                label=label,
                title=f"{node_type}: {label}",
                color=color,
                size=size,
                group=node_type
            )

        # 添加边
        node_ids = {n['id'] for n in nodes}
        for edge in edges:
            # Skip edges with non-existent nodes
            if edge['source'] not in node_ids or edge['target'] not in node_ids:
                continue

            rel_type = edge.get('relation', 'connected').upper()
            title = f"{rel_type}"
            if 'desc' in edge:
                title += f"\n{edge['desc']}"

            net.add_edge(
                edge['source'],
                edge['target'],
                title=title,
                label=rel_type,
                color='gray' if rel_type == 'PART_OF' else 'white'
            )

        # 保存 HTML
        output_path = Path(output_html).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        net.save_graph(str(output_path))

        # Add interactive filter UI
        self._add_filter_ui(str(output_path), nodes)

        logger.info(f"可视化已保存: {output_path}")

        return str(output_path)

    # ---------------------------------------------------------------------
    # 从 Neo4j 可视化
    # ---------------------------------------------------------------------

    def visualize_from_neo4j(
        self,
        builder,
        cypher_query: str,
        output_html: str = "output/neo4j_visualization.html",
        title: str = "Neo4j Graph Visualization"
    ) -> str:
        """
        从 Neo4j 数据库生成可视化

        Args:
            builder: Neo4jBuilder 实例
            cypher_query: 查询节点的 Cypher 语句
            output_html: 输出 HTML 文件路径
            title: 图谱标题

        Returns:
            输出 HTML 文件路径

        示例查询:
            "MATCH (n:Creature)-[r]-(m) RETURN n, r, m LIMIT 50"
        """
        logger.info("从 Neo4j 读取数据...")

        with builder._driver.session(database=builder.database) as session:
            result = session.run(cypher_query)

            # 收集节点和边
            nodes = {}
            edges = []

            for record in result:
                # 处理节点
                for key in record:
                    value = record[key]
                    if hasattr(value, 'element_type') and value.element_type == 'node':
                        node_id = value.element_id
                        if node_id not in nodes:
                            nodes[node_id] = {
                                'id': value.get('id', node_id),
                                'label': value.get('label', ''),
                                'labels': list(value.labels)
                            }
                    elif hasattr(value, 'element_type') and value.element_type == 'relationship':
                        edges.append({
                            'source': value.start_node.element_id,
                            'target': value.end_node.element_id,
                            'relation': value.type
                        })

        logger.info(f"读取到 {len(nodes)} 节点, {len(edges)} 边")

        # 创建网络
        net = self._create_pyvis_network(title)

        # 添加节点
        for node in nodes.values():
            node_type = node['labels'][0] if node['labels'] else 'Entity'
            color = self.DEFAULT_COLORS.get(node_type, self.DEFAULT_COLORS['default'])
            label = node.get('label', node['id'])

            net.add_node(
                node['id'],
                label=label,
                title=f"{node_type}: {label}",
                color=color,
                group=node_type
            )

        # 添加边
        for edge in edges:
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge['relation'],
                label=edge['relation']
            )

        # 保存
        output_path = Path(output_html).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))

        logger.info(f"可视化已保存: {output_path}")
        return str(output_path)

    # ---------------------------------------------------------------------
    # 联合可视化（位置 + 实体）
    # ---------------------------------------------------------------------

    def visualize_combined(
        self,
        location_graph_file: str,
        entity_graph_file: str,
        output_html: str = "output/combined_visualization.html",
        max_nodes: int = 200
    ) -> str:
        """
        合并位置图谱和实体图谱的可视化

        Args:
            location_graph_file: location_graph.json 路径
            entity_graph_file: entity_graph.json 路径
            output_html: 输出 HTML 文件路径
            max_nodes: 最大节点数

        Returns:
            输出 HTML 文件路径
        """
        logger.info("生成联合可视化...")

        # 读取两个图谱
        with open(location_graph_file, encoding='utf-8') as f:
            loc_data = json.load(f)
        with open(entity_graph_file, encoding='utf-8') as f:
            ent_data = json.load(f)

        # 合并节点（去重，处理 ID 前缀问题）
        all_nodes = {}
        # Track ID mappings: prefixed_id -> base_id for locations
        id_mapping = {}  # entity_graph_id -> actual_key_in_all_nodes

        # 首先添加位置图谱的节点
        for node in loc_data.get('nodes', []):
            all_nodes[node['id']] = {
                'id': node['id'],
                'label': node['label'],
                'type': 'Location'
            }

        # 添加实体图谱的节点，处理 location: 前缀
        for node in ent_data.get('nodes', []):
            node_id = node['id']
            node_type = node.get('type', 'Entity').lower()

            # 如果是位置类型，尝试匹配已存在的位置节点
            if node_type == 'location' or node_type == 'loc':
                # 移除 location: 前缀进行匹配
                base_id = node_id.replace('location:', '')
                if base_id in all_nodes:
                    # 已存在，记录映射关系（实体图谱中的 ID -> 位置图谱中的 ID）
                    id_mapping[node_id] = base_id
                    continue

            # 添加新节点
            all_nodes[node_id] = {
                'id': node_id,
                'label': node['label'],
                'type': node.get('type', 'Entity')
            }

        # 合并边（规范化节点 ID）
        all_edges = []

        def normalize_id(node_id: str, nodes_dict: dict) -> str:
            """规范化节点 ID，处理 location: 前缀"""
            # First check if we have an explicit mapping
            if node_id in id_mapping:
                return id_mapping[node_id]
            # Check if ID exists directly
            if node_id in nodes_dict:
                return node_id
            # Try removing common prefixes
            for prefix in ['location:', 'creature:', 'item:', 'spell:', 'party:']:
                if node_id.startswith(prefix):
                    base_id = node_id.replace(prefix, '', 1)
                    if base_id in nodes_dict:
                        return base_id
                    # Check mapping for base_id
                    if base_id in id_mapping:
                        return id_mapping[base_id]
            # If not found, return original (will cause edge to be filtered)
            return node_id

        for edge in loc_data.get('edges', []):
            all_edges.append(edge)

        for edge in ent_data.get('edges', []):
            # 规范化源和目标 ID
            source = normalize_id(edge['source'], all_nodes)
            target = normalize_id(edge['target'], all_nodes)
            # 只添加两端都存在的边
            if source in all_nodes and target in all_nodes:
                all_edges.append({
                    'source': source,
                    'target': target,
                    'relation': edge.get('relation', 'connected'),
                    'desc': edge.get('desc', '')
                })

        # 采样
        if len(all_nodes) > max_nodes:
            # 简单策略：保留实体图谱中的节点
            with open(entity_graph_file, encoding='utf-8') as f:
                ent_data = json.load(f)
            keep_ids = {n['id'] for n in ent_data.get('nodes', [])}
            # Also keep any base IDs that are mapped from kept IDs
            for entity_id in list(keep_ids):
                if entity_id in id_mapping:
                    keep_ids.add(id_mapping[entity_id])
            all_nodes = {k: v for k, v in all_nodes.items() if k in keep_ids}
            # Filter edges based on actual keys remaining in all_nodes
            all_edges = [
                e for e in all_edges
                if e['source'] in all_nodes and e['target'] in all_nodes
            ]

        # 创建网络
        net = self._create_pyvis_network("LoreWeaver: Combined Knowledge Graph")

        # 添加节点
        for node in all_nodes.values():
            node_type = node['type'].title()
            color = self.DEFAULT_COLORS.get(node_type, self.DEFAULT_COLORS['default'])
            label = node['label']

            net.add_node(
                node['id'],
                label=label,
                title=f"{node_type}: {label}",
                color=color,
                group=node_type
            )

        # 添加边
        for edge in all_edges:
            rel_type = edge.get('relation', 'connected').upper()
            title = rel_type
            if 'desc' in edge:
                title += f"\n{edge['desc']}"

            net.add_edge(
                edge['source'],
                edge['target'],
                title=title,
                label=rel_type
            )

        # 保存
        output_path = Path(output_html).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))

        logger.info(f"联合可视化已保存: {output_path}")
        return str(output_path)

    # ---------------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------------

    def _create_pyvis_network(self, title: str) -> Network:
        """创建配置好的 Pyvis 网络对象"""
        net = Network(
            width=self.width,
            height=self.height,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=False,
            notebook=False
        )

        # 统一配置选项（使用 JSON 格式）
        if self.physics:
            options = {
                "physics": {
                    "enabled": True,
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.04
                    }
                },
                "title": title,
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 200,
                    "hoverConnectedEdges": True
                },
                "nodes": {
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "chosen": True
                },
                "edges": {
                    "width": 1,
                    "selectionWidth": 2,
                    "smooth": {
                        "type": "continuous"
                    }
                }
            }
        else:
            options = {
                "physics": {"enabled": False},
                "title": title,
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 200,
                    "hoverConnectedEdges": True
                },
                "nodes": {
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "chosen": True
                },
                "edges": {
                    "width": 1,
                    "selectionWidth": 2
                }
            }

        net.set_options(json.dumps(options))

        return net

    def _add_filter_ui(self, html_path: str, nodes: list[dict]) -> None:
        """
        Inject interactive filter UI and highlighting into the HTML file.

        Adds:
        1. Filter panel with checkboxes for each node type
        2. Interactive highlighting on node click
        3. Show/hide functionality for nodes by type
        """
        # Collect unique node types
        node_types = set()
        for node in nodes:
            node_type = node.get('type', node.get('node_type', 'Entity'))
            node_types.add(node_type.title() if node_type else 'Unknown')

        sorted_types = sorted(node_types)
        type_colors = {t: self.DEFAULT_COLORS.get(t, self.DEFAULT_COLORS['default']) for t in sorted_types}

        # Read the existing HTML
        with open(html_path, encoding='utf-8') as f:
            html_content = f.read()

        # Custom CSS and JavaScript to inject
        custom_css = """
        <style>
        /* Filter Panel Styles */
        #filter-panel {
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(34, 34, 34, 0.95);
            border: 1px solid #555;
            border-radius: 8px;
            padding: 15px;
            z-index: 1000;
            min-width: 200px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }

        #filter-panel h3 {
            margin: 0 0 10px 0;
            color: #fff;
            font-size: 14px;
            border-bottom: 1px solid #555;
            padding-bottom: 8px;
        }

        #filter-panel .filter-item {
            display: flex;
            align-items: center;
            margin: 6px 0;
            cursor: pointer;
        }

        #filter-panel .filter-item input {
            margin-right: 8px;
            cursor: pointer;
        }

        #filter-panel .filter-item label {
            color: #ccc;
            cursor: pointer;
            flex: 1;
            display: flex;
            align-items: center;
        }

        #filter-panel .color-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 6px;
        }

        #filter-panel .filter-buttons {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid #555;
        }

        #filter-panel button {
            flex: 1;
            padding: 6px 12px;
            background: #444;
            color: #fff;
            border: 1px solid #666;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        #filter-panel button:hover {
            background: #555;
        }

        #filter-panel button.active {
            background: #3498db;
            border-color: #3498db;
        }

        #node-info-panel {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: rgba(34, 34, 34, 0.95);
            border: 1px solid #555;
            border-radius: 8px;
            padding: 15px;
            z-index: 1000;
            min-width: 250px;
            max-width: 400px;
            display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }

        #node-info-panel h3 {
            margin: 0 0 10px 0;
            color: #fff;
            font-size: 14px;
            border-bottom: 1px solid #555;
            padding-bottom: 8px;
        }

        #node-info-panel .info-row {
            margin: 6px 0;
            color: #ccc;
            font-size: 13px;
        }

        #node-info-panel .info-label {
            color: #888;
            margin-right: 8px;
        }

        #node-info-panel .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 18px;
            padding: 0;
            width: 24px;
            height: 24px;
        }

        #node-info-panel .close-btn:hover {
            color: #fff;
        }

        #stats-panel {
            position: fixed;
            top: 10px;
            left: 10px;
            background: rgba(34, 34, 34, 0.95);
            border: 1px solid #555;
            border-radius: 8px;
            padding: 12px 15px;
            z-index: 1000;
            color: #ccc;
            font-size: 13px;
        }

        #stats-panel .stat-item {
            margin: 4px 0;
        }

        #stats-panel .stat-value {
            color: #3498db;
            font-weight: bold;
            margin-left: 8px;
        }
        </style>
        """

        custom_js = f"""
        <script>
        // Store enhancement state in window object to avoid scope issues
        window.lwEnhancement = {{}};

        // Node type colors (must match Python DEFAULT_COLORS)
        window.lwEnhancement.typeColors = {{ {', '.join([f"'{t}': '{c}'" for t, c in type_colors.items()])} }};

        // Original nodes data (stored for filtering)
        window.lwEnhancement.originalNodes = null;
        window.lwEnhancement.originalEdges = null;
        window.lwEnhancement.selectedNodeId = null;

        // Get network reference and store original data
        window.lwEnhancement.init = function() {{
            // Get the global network variable created by pyvis
            // pyvis creates it as 'var network = new vis.Network(...)'
            let net = null;

            // Try different ways to access the network
            if (typeof network !== 'undefined' && network && typeof network.body === 'object') {{
                net = network;
                console.log("Found network via global variable");
            }} else if (window.network && typeof window.network.body === 'object') {{
                net = window.network;
                console.log("Found network via window.network");
            }}

            if (!net) {{
                console.error("Could not find network object. Available globals:", Object.keys(window).filter(k => k.includes('net')));
                return;
            }}

            // Store reference
            window.lwEnhancement.network = net;

            // Store original nodes and edges data
            const nodesData = net.body.data.nodes;
            const edgesData = net.body.data.edges;

            window.lwEnhancement.originalNodes = nodesData.get({{ returnType: "Object" }});
            window.lwEnhancement.originalEdges = edgesData.get({{ returnType: "Object" }});

            console.log("Network initialized with", Object.keys(window.lwEnhancement.originalNodes).length, "nodes");

            // Setup click event for highlighting
            net.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    window.lwEnhancement.highlightNode(params.nodes[0]);
                }} else {{
                    window.lwEnhancement.clearHighlight();
                }}
            }});

            // Setup double-click to zoom to node
            net.on("doubleClick", function(params) {{
                if (params.nodes.length > 0) {{
                    net.focus(params.nodes[0], {{ scale: 1.5, animation: true }});
                }}
            }});

            window.lwEnhancement.updateStats();
        }};

        // Highlight selected node and its connections
        window.lwEnhancement.highlightNode = function(nodeId) {{
            window.lwEnhancement.selectedNodeId = nodeId;
            const net = window.lwEnhancement.network;
            if (!net) return;

            const allNodes = net.body.data.nodes.get();
            const allEdges = net.body.data.edges.get();

            // Get connected nodes and edges
            const connected = net.getConnectedNodes(nodeId);
            const connectedEdgeIds = [];

            allEdges.forEach(edge => {{
                if (edge.from === nodeId || edge.to === nodeId) {{
                    connectedEdgeIds.push(edge.id);
                }}
            }});

            // Dim all nodes first
            const updateNodes = allNodes.map(node => {{
                const isConnected = connected.includes(node.id) || node.id === nodeId;
                return {{
                    ...node,
                    opacity: isConnected ? 1 : 0.2,
                    borderWidth: node.id === nodeId ? 4 : (isConnected ? 3 : 1),
                    size: node.id === nodeId ? (node.size || 20) * 1.3 : (isConnected ? (node.size || 15) * 1.1 : (node.size || 15))
                }};
            }});

            // Highlight connected edges
            const updateEdges = allEdges.map(edge => {{
                const isHighlighted = connectedEdgeIds.includes(edge.id);
                return {{
                    ...edge,
                    opacity: isHighlighted ? 1 : 0.1,
                    width: isHighlighted ? 2 : 1
                }};
            }});

            net.body.data.nodes.update(updateNodes);
            net.body.data.edges.update(updateEdges);

            // Show node info panel
            window.lwEnhancement.showNodeInfo(nodeId, connected);
        }};

        // Clear highlight and restore all nodes
        window.lwEnhancement.clearHighlight = function() {{
            window.lwEnhancement.selectedNodeId = null;
            const net = window.lwEnhancement.network;
            const originalNodes = window.lwEnhancement.originalNodes;

            if (!net || !originalNodes) return;

            const allNodes = net.body.data.nodes.get();
            const allEdges = net.body.data.edges.get();

            const restoreNodes = allNodes.map(node => {{
                const original = originalNodes[node.id];
                return {{
                    ...node,
                    opacity: 1,
                    borderWidth: original ? original.borderWidth : 2,
                    size: original ? original.size : node.size
                }};
            }});

            const restoreEdges = allEdges.map(edge => {{
                return {{
                    ...edge,
                    opacity: 1,
                    width: 1
                }};
            }});

            net.body.data.nodes.update(restoreNodes);
            net.body.data.edges.update(restoreEdges);

            // Hide node info panel
            document.getElementById('node-info-panel').style.display = 'none';
        }};

        // Show node info panel
        window.lwEnhancement.showNodeInfo = function(nodeId, connectedNodes) {{
            const net = window.lwEnhancement.network;
            if (!net) return;

            const node = net.body.data.nodes.get(nodeId);
            const panel = document.getElementById('node-info-panel');

            let html = '<button class="close-btn" onclick="document.getElementById(\\'node-info-panel\\').style.display=\\'none\\'">&times;</button>';
            html += `<h3>${{node.label || nodeId}}</h3>`;
            html += `<div class="info-row"><span class="info-label">ID:</span>${{nodeId}}</div>`;
            html += `<div class="info-row"><span class="info-label">Type:</span>${{node.group || 'Unknown'}}</div>`;
            html += `<div class="info-row"><span class="info-label">Connections:</span>${{connectedNodes.length}}</div>`;

            if (connectedNodes.length > 0) {{
                html += '<div class="info-row" style="margin-top:10px;"><span class="info-label">Connected to:</span></div>';
                connectedNodes.slice(0, 10).forEach(connId => {{
                    const connNode = net.body.data.nodes.get(connId);
                    html += `<div class="info-row" style="margin-left:10px;">• ${{connNode ? connNode.label : connId}}</div>`;
                }});
                if (connectedNodes.length > 10) {{
                    html += `<div class="info-row" style="margin-left:10px; color:#888;">... and ${{connectedNodes.length - 10}} more</div>`;
                }}
            }}

            panel.innerHTML = html;
            panel.style.display = 'block';
        }};

        // Filter nodes by type
        window.lwEnhancement.filterByType = function(selectedTypes) {{
            const net = window.lwEnhancement.network;
            const originalNodes = window.lwEnhancement.originalNodes;

            if (!net || !originalNodes) return;

            const allEdges = net.body.data.edges.get();

            // Get visible node IDs based on selected types
            const visibleNodeIds = Object.keys(originalNodes).filter(id => {{
                const node = originalNodes[id];
                const nodeType = node.group || 'Unknown';
                return selectedTypes.includes(nodeType);
            }});

            console.log("Filtering by types:", selectedTypes, "visible nodes:", visibleNodeIds.length);

            // Filter nodes: only show selected types
            const filteredNodes = Object.values(originalNodes).map(node => ({{
                ...node,
                hidden: !visibleNodeIds.includes(node.id)
            }}));

            // Filter edges: only show edges between visible nodes
            const visibleNodeIdSet = new Set(visibleNodeIds);
            const filteredEdges = allEdges.map(edge => ({{
                ...edge,
                hidden: !(visibleNodeIdSet.has(edge.from) && visibleNodeIdSet.has(edge.to))
            }}));

            net.body.data.nodes.update(filteredNodes);
            net.body.data.edges.update(filteredEdges);

            // Clear any highlight
            window.lwEnhancement.clearHighlight();
            window.lwEnhancement.updateStats(visibleNodeIds.length);
        }};

        // Show all nodes
        window.lwEnhancement.showAllTypes = function() {{
            const net = window.lwEnhancement.network;
            const originalNodes = window.lwEnhancement.originalNodes;

            if (!net || !originalNodes) return;

            const filteredNodes = Object.values(originalNodes).map(node => ({{
                ...node,
                hidden: false
            }}));

            const filteredEdges = net.body.data.edges.get().map(edge => ({{
                ...edge,
                hidden: false
            }}));

            net.body.data.nodes.update(filteredNodes);
            net.body.data.edges.update(filteredEdges);

            window.lwEnhancement.clearHighlight();
            window.lwEnhancement.updateStats(Object.keys(originalNodes).length);
        }};

        // Update stats display
        window.lwEnhancement.updateStats = function(visibleCount) {{
            const originalNodes = window.lwEnhancement.originalNodes;
            const originalEdges = window.lwEnhancement.originalEdges;

            const totalNodes = originalNodes ? Object.keys(originalNodes).length : 0;
            const totalEdges = originalEdges ? Object.keys(originalEdges).length : 0;
            const count = visibleCount !== null ? visibleCount : totalNodes;

            document.getElementById('stats-visible-nodes').textContent = count;
            document.getElementById('stats-total-nodes').textContent = totalNodes;
            document.getElementById('stats-total-edges').textContent = totalEdges;
        }};

        // Initialize after page load
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', function() {{
                setTimeout(window.lwEnhancement.init, 100);
            }});
        }} else {{
            setTimeout(window.lwEnhancement.init, 100);
        }}
        </script>
        """

        # Global functions for filter buttons (need to be global for onclick attributes)
        filter_js = """
        <script>
        function toggleFilter(type) {{
            const checkboxes = document.querySelectorAll('.type-checkbox');
            const selected = [];
            checkboxes.forEach(cb => {{
                if (cb.checked) selected.push(cb.value);
            }});

            console.log("Selected types:", selected);

            if (selected.length === 0) {{
                // If nothing selected, show all
                window.lwEnhancement.showAllTypes();
            }} else {{
                window.lwEnhancement.filterByType(selected);
            }}
        }}

        function selectAllFilters() {{
            document.querySelectorAll('.type-checkbox').forEach(cb => cb.checked = true);
            window.lwEnhancement.showAllTypes();
        }}

        function clearAllFilters() {{
            document.querySelectorAll('.type-checkbox').forEach(cb => cb.checked = false);
            window.lwEnhancement.showAllTypes();
        }}
        </script>
        """

        # Global functions for filter buttons (need to be global for onclick attributes)
        filter_js = """
        <script>
        function toggleFilter(type) {{
            const checkboxes = document.querySelectorAll('.type-checkbox');
            const selected = [];
            checkboxes.forEach(cb => {{
                if (cb.checked) selected.push(cb.value);
            }});

            console.log("Selected types:", selected);

            if (selected.length === 0) {{
                // If nothing selected, show all
                window.lwEnhancement.showAllTypes();
            }} else {{
                window.lwEnhancement.filterByType(selected);
            }}
        }}

        function selectAllFilters() {{
            document.querySelectorAll('.type-checkbox').forEach(cb => cb.checked = true);
            window.lwEnhancement.showAllTypes();
        }}

        function clearAllFilters() {{
            document.querySelectorAll('.type-checkbox').forEach(cb => cb.checked = false);
            window.lwEnhancement.showAllTypes();
        }}
        </script>
        """

        # HTML for filter panel
        filter_html = """
        <div id="filter-panel">
            <h3>🔍 Filter by Type</h3>
            <div id="filter-items">
        """

        for node_type in sorted_types:
            color = type_colors.get(node_type, '#bdc3c7')
            filter_html += f"""
                <div class="filter-item">
                    <input type="checkbox" class="type-checkbox" value="{node_type}" checked onchange="toggleFilter()">
                    <label>
                        <span class="color-dot" style="background: {color}"></span>
                        {node_type}
                    </label>
                </div>
            """

        filter_html += """
            </div>
            <div class="filter-buttons">
                <button onclick="selectAllFilters()">All</button>
                <button onclick="clearAllFilters()">None</button>
            </div>
        </div>

        <div id="node-info-panel">
        </div>

        <div id="stats-panel">
            <div class="stat-item">
                Visible Nodes: <span class="stat-value" id="stats-visible-nodes">-</span>
            </div>
            <div class="stat-item">
                Total Nodes: <span class="stat-value" id="stats-total-nodes">-</span>
            </div>
            <div class="stat-item">
                Total Edges: <span class="stat-value" id="stats-total-edges">-</span>
            </div>
        </div>
        """

        # Inject CSS, JS, and HTML into the file
        # Insert CSS after <head>
        html_content = html_content.replace('</head>', custom_css + '</head>')

        # Insert JS before </body> (both custom_js and filter_js)
        html_content = html_content.replace('</body>', custom_js + filter_js + '</body>')

        # Insert filter HTML after <body>
        html_content = html_content.replace('<body>', '<body>' + filter_html)

        # Write the modified HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

if __name__ == "__main__":
    visualizer = GraphVisualizer()

    # 1. 从 JSON 可视化位置图谱
    visualizer.visualize_from_json(
        "output/location_graph.json",
        output_html="output/location_viz.html",
        title="Location Graph - Stormwreck Isle"
    )

    # 2. 从 JSON 可视化实体图谱
    visualizer.visualize_from_json(
        "output/entity_graph.json",
        output_html="output/entity_viz.html",
        node_filter=['Creature', 'Location'],  # 只显示生物和位置
        max_nodes=100,
        title="Entity Graph - Creatures & Locations"
    )

    # 3. 联合可视化
    visualizer.visualize_combined(
        "output/location_graph.json",
        "output/entity_graph.json",
        output_html="output/combined_viz.html"
    )

    # 4. 从 Neo4j 可视化（需要先建图）
    # from .neo4j_builder import Neo4jBuilder
    # builder = Neo4jBuilder(uri="bolt://localhost:7687", user="neo4j", password="password")
    # with builder:
    #     visualizer.visualize_from_neo4j(
    #         builder,
    #         "MATCH (n:Creature)-[r]-(m) RETURN n, r, m LIMIT 50",
    #         output_html="output/neo4j_creature_viz.html"
    #     )
