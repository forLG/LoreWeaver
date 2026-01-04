import streamlit as st
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import sys
import time
import html
import re
import subprocess
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置页面配置
st.set_page_config(page_title="LoreWeaver Holodeck", layout="wide", page_icon="🐉")

# --- 样式美化 (自定义 CSS) ---
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    h1 { color: #2c3e50; }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        width: 100%;
    }
    .stSelectbox {
        color: #333333;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐉 LoreWeaver: Dungeon Master's Console")

# --- 侧边栏：控制面板 ---
with st.sidebar:
    st.header("Pipeline Controls")
    
    # 模型选择
    model_choice = st.selectbox(
        "Model", 
        ["deepseek-chat", "qwen3-8b"], 
        index=0,
        help="Select the LLM model to use for processing."
    )
    
    # 阶段选择
    stages = st.multiselect(
        "Stages", 
        ["shadow", "spatial", "section-map", "entity"], 
        default=["entity"],
        help="Select pipeline stages to run."
    )
    
    # 并发控制
    concurrency = st.slider("Concurrency", min_value=1, max_value=50, value=5)
    
    st.divider()
    
    # 文件上传
    uploaded_file = st.file_uploader("Upload Adventure JSON", type=['json'])
    input_file_path = None
    
    if uploaded_file is not None:
        # 保存上传的文件到 data 目录
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        input_file_path = data_dir / uploaded_file.name
        with open(input_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name} to data/")
    
    st.divider()
    
    if st.button("🚀 Run Pipeline"):
        st.toast(f"Starting pipeline with {model_choice}...", icon="🔥")
        
        with st.status("Processing...", expanded=True) as status:
            st.write("Initializing pipeline...")
            
            # 构建命令
            cmd = [sys.executable, "-m", "src.main"]
            
            # 强制按照正确的依赖顺序排序 Stages
            # 无论用户在前端选择的顺序如何，执行顺序必须是：shadow -> spatial -> section-map -> entity
            ORDERED_STAGES = ["shadow", "spatial", "section-map", "entity"]
            sorted_stages = [s for s in ORDERED_STAGES if s in stages]
            
            for stage in sorted_stages:
                cmd.extend(["--stage", stage])
            
            # 如果有上传文件，指定 input 参数
            if input_file_path:
                cmd.extend(["--input", str(input_file_path)])
            
            # 传递参数
            # 同时设置环境变量以确保兼容性
            env = os.environ.copy()
            env["LLM_MODEL"] = model_choice
            env["LLM_MAX_CONCURRENT"] = str(concurrency)
            
            # 显式传递命令行参数
            cmd.extend(["--max-concurrent", str(concurrency)])
            # 注意：main.py 可能不直接接受 --model 参数，而是依赖环境变量或 config
            # 但为了保险，我们只依赖环境变量传递模型选择，除非确定 main.py 有 --model 参数
            # 根据之前的分析，main.py 似乎没有 --model 参数，而是通过 get_default_concurrency 读取 env
            # 等等，之前的 read_file 显示 main.py 的 docstring 里有 python -m main ... --model qwen3-8b
            # 让我们假设它支持，如果不支持，argparse 会报错。
            # 为了安全，我们先不传 --model 参数，而是完全依赖环境变量 LLM_MODEL
            # 再次检查 main.py... parse_args 里没有 --model。
            # 修正：main.py 的 parse_args 里确实没有 --model。
            # 它是在 get_default_concurrency 里读取 os.getenv('LLM_MODEL')。
            # 所以我们只设置环境变量。
            
            st.code(" ".join(cmd), language="bash")
            
            log_container = st.empty()
            output_lines = []
            
            try:
                # 使用 subprocess.Popen 实时执行
                process = subprocess.Popen(
                    cmd,
                    cwd=str(project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding='utf-8',
                    env=env
                )
                
                # 实时读取输出
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        line = line.strip()
                        if line: # 忽略空行
                            output_lines.append(line)
                            # 滚动显示最后 15 行日志
                            log_container.code("\n".join(output_lines[-15:]), language="bash")
                
                if process.returncode == 0:
                    status.update(label="Pipeline Complete!", state="complete", expanded=False)
                    st.success("Task finished successfully!")
                    # 清除数据缓存，以便重新加载最新的图谱
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun() # 刷新页面
                else:
                    status.update(label="Pipeline Failed!", state="error", expanded=True)
                    st.error(f"Process failed with return code {process.returncode}")
                    
            except Exception as e:
                st.error(f"Failed to run pipeline: {e}")

    st.divider()
    st.header("Graph Settings")
    physics = st.checkbox("Enable Physics", value=True)
    graph_source = st.selectbox(
        "Graph Source",
        ["Entity Graph", "Location Graph", "Shadow Tree"],
        index=0
    )

# --- 主界面：图谱可视化 ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Interactive View: {graph_source}")
    
    def get_graph_path(source_name, model_name):
        base_dir = project_root / "output"
        # 简单的映射逻辑，根据实际输出目录结构调整
        model_dir = "deepseek" if "deepseek" in model_name else "qwen3"
        
        if source_name == "Entity Graph":
            return base_dir / model_dir / "entity_graph.json"
        elif source_name == "Location Graph":
            return base_dir / model_dir / "location_graph.json"
        elif source_name == "Shadow Tree":
            return base_dir / model_dir / "shadow_tree.json"
        return None

    @st.cache_data
    def load_graph_data(file_path):
        if file_path and file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
        return None

    # 1. 加载数据
    target_path = get_graph_path(graph_source, model_choice)
    graph_data = load_graph_data(target_path)

    # 2. 提取类型并显示过滤器
    all_types = []
    if graph_data and "nodes" in graph_data:
        # 使用 title() 统一格式，避免 "Location" 和 "location" 分开
        all_types = sorted(list(set((n.get('type') or 'Unknown').title() for n in graph_data['nodes'])))
    
    selected_types = []
    if all_types:
        with st.sidebar:
            st.subheader("Highlight Nodes")
            selected_types = st.multiselect(
                "Highlight Node Types", 
                all_types, 
                default=all_types
            )
    
    def visualize_graph(data, use_physics, filter_types=None):
        net = Network(height="600px", width="100%", bgcolor="#f0f2f6", font_color="#333333")
        
        if data:
            try:
                node_count = 0
                edge_count = 0
                
                if "nodes" in data and "edges" in data:
                    # 标准图结构
                    existing_nodes = set()
                    highlighted_ids = set() # 记录高亮的节点ID

                    # First pass: Calculate degrees
                    degrees = {}
                    for edge in data['edges']:
                        source = edge['source']
                        target = edge['target']
                        degrees[source] = degrees.get(source, 0) + 1
                        degrees[target] = degrees.get(target, 0) + 1

                    # Second pass: Add all nodes, applying dimming if needed
                    for node in data['nodes']:
                        if node_count > 1000: break
                        
                        raw_type = node.get('type') or 'Unknown'
                        normalized_type = raw_type.title()
                        
                        # 判断是否高亮
                        is_highlighted = True
                        if filter_types is not None and normalized_type not in filter_types:
                            is_highlighted = False

                        node_id = node['id']
                        label = node.get('label') or node.get('name') or node_id
                        title = node.get('desc', '') or str(node)
                        
                        # 计算节点大小 (基础大小 25 + 度数 * 3) - 调大基础大小以容纳文字
                        # 限制最大大小以防过大
                        node_degree = degrees.get(node_id, 0)
                        size = 25 + min(node_degree * 3, 60)
                        
                        # 默认颜色逻辑 (User Custom Pastel Palette)
                        color = '#CCCCCC' # Default
                        
                        if 'Location' in normalized_type: 
                            color = '#CCFFFF' # Light Cyan
                        elif 'Person' in normalized_type or 'Creature' in normalized_type: 
                            color = '#FF6666' # Pastel Red
                        elif 'Party' in normalized_type: 
                            color = '#CCFF99' # Pastel Green
                        elif 'Spell' in normalized_type: 
                            color = '#CCCCFF' # Pastel Purple
                        elif 'Item' in normalized_type: 
                            color = '#FFCC99' # Pastel Orange
                        elif 'Event' in normalized_type:
                            color = '#FFFF99' # Pastel Yellow
                        
                        # 如果不高亮，则变暗
                        if not is_highlighted:
                            color = 'rgba(50, 50, 50, 0.1)' # 更加透明的深灰色
                        else:
                            highlighted_ids.add(node_id)
                        
                        # --- 1. 美化 Tooltip (HTML) ---
                        desc = str(node.get('desc', ''))
                        if not desc or desc.lower() == 'no description available.':
                            desc = ''
                        
                        # 安全处理：转义内容中的特殊字符，但保留 HTML 结构标签
                        safe_label = html.escape(label)
                        safe_type = html.escape(normalized_type)
                        
                        # 构建纯文本 Tooltip
                        # 回归简单文本，避免 HTML 渲染问题
                        tooltip_text = f"【{label}】\n[{normalized_type}]"
                        
                        if desc:
                            tooltip_text += f"\n\n{desc}"

                        # --- 2. 标签截断逻辑 ---
                        # 根据圆圈大小估算能放多少字
                        max_chars = max(4, int(size / 3.5))
                        display_label = label
                        if len(label) > max_chars:
                            display_label = label[:max_chars] + ".."

                        # --- 3. 字体样式 ---
                        # 深蓝灰字体 + 白色描边，确保在粉彩背景上清晰可见且不使用纯黑
                        font_style = {
                            'color': '#2c3e50', 
                            'size': 14, 
                            'face': 'arial',
                            'strokeWidth': 3,
                            'strokeColor': '#ffffff',
                            'bold': True,
                            'vadjust': 0 # 垂直居中
                        }

                        net.add_node(
                            node_id, 
                            label=display_label, 
                            title=tooltip_text, 
                            color=color, 
                            size=size,
                            shape='circle', # 圆形，让文字显示在内部
                            font=font_style
                        )
                        existing_nodes.add(node_id)
                        node_count += 1
                        
                    # Third pass: Add edges
                    for edge in data['edges']:
                        if edge_count > 2000: break
                        
                        source = edge['source']
                        target = edge['target']
                        
                        # 自动补全缺失节点（默认为暗色，小尺寸）
                        if source not in existing_nodes:
                            net.add_node(source, label=source, color='rgba(50, 50, 50, 0.3)', title="Auto-generated", size=10)
                            existing_nodes.add(source)
                        
                        if target not in existing_nodes:
                            net.add_node(target, label=target, color='rgba(50, 50, 50, 0.3)', title="Auto-generated", size=10)
                            existing_nodes.add(target)

                        try:
                            # 如果连接的任一节点未高亮，则边也变暗
                            edge_color = None
                            if source not in highlighted_ids or target not in highlighted_ids:
                                edge_color = 'rgba(50, 50, 50, 0.1)' # 几乎隐形的边
                                
                            net.add_edge(source, target, title=edge.get('relation', ''), color=edge_color)
                            edge_count += 1
                        except Exception as e:
                            print(f"Skipping edge {source} -> {target}: {e}")
                            continue
                        
                elif isinstance(data, list):
                    # 可能是 Shadow Tree 的列表结构
                    st.warning("Tree visualization not fully implemented yet. Showing raw structure.")
                    pass
                
                st.caption(f"Displaying {node_count} nodes and {edge_count} edges")
                
            except Exception as e:
                st.error(f"Error visualizing graph: {e}")
                return None
        else:
            st.info(f"No data found or file missing: {target_path}")
            # Demo Graph
            net.add_node(1, label="Town Square", color="#ff5733", title="The center of the village.")
            net.add_node(2, label="Tavern", color="#33ff57", title="A lively place.")
            net.add_node(3, label="Blacksmith", color="#3357ff", title="Clang clang.")
            net.add_edge(1, 2)
            net.add_edge(1, 3)

        if use_physics:
            # 启用物理引擎并配置参数以避免节点重叠
            # 调整参数以减少震荡：增加阻尼(damping)，适度斥力
            net.set_options("""
            {
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -2000,
                  "centralGravity": 0.1,
                  "springLength": 120,
                  "springConstant": 0.04,
                  "damping": 0.9,
                  "avoidOverlap": 0.5
                },
                "minVelocity": 0.75,
                "stabilization": {
                    "enabled": true,
                    "iterations": 200
                }
              }
            }
            """)
        else:
            net.toggle_physics(False)
        
        # 保存到临时文件
        # 使用绝对路径确保 Streamlit 能找到
        tmp_path = project_root / "frontend" / "temp_graph.html"
        net.save_graph(str(tmp_path))
        return tmp_path

    html_path = visualize_graph(graph_data, physics, selected_types)

    
    if html_path and html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
            # --- 注入自定义 CSS 美化 Tooltip ---
            # 即使是纯文本，也可以通过 CSS 美化容器
            custom_css = """
            <style>
                div.vis-tooltip {
                    background-color: rgba(255, 255, 255, 0.98) !important;
                    color: #333 !important;
                    border: 1px solid #e0e0e0 !important;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.15) !important;
                    padding: 12px 16px !important;
                    border-radius: 8px !important;
                    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
                    font-size: 14px !important;
                    line-height: 1.5 !important;
                    max-width: 320px !important;
                    white-space: pre-wrap !important; /* 保留换行符 */
                    z-index: 99999 !important;
                }
            </style>
            """
            source_code = source_code.replace("</head>", f"{custom_css}</head>")
            
        components.html(source_code, height=610)

# --- 右侧栏：详细信息 ---
with col2:
    st.subheader("Entity Inspector")
    
    # 创建一个字典用于快速查找节点信息
    node_lookup = {}
    if graph_data and "nodes" in graph_data:
        for node in graph_data['nodes']:
            # 建立多重索引：ID 和 Label 都可以查
            node_lookup[node['id']] = node
            if 'label' in node:
                node_lookup[node['label']] = node
            if 'name' in node:
                node_lookup[node['name']] = node

    # 获取所有可选的节点名称用于搜索建议
    search_options = sorted(list(node_lookup.keys()))

    selected_entity_id = st.selectbox(
        "Select or Search Entity",
        options=[""] + search_options,
        format_func=lambda x: x if x else "Type to search...",
        help="Select a node to view details."
    )

    if selected_entity_id and selected_entity_id in node_lookup:
        node = node_lookup[selected_entity_id]
        
        # 展示详细信息卡片
        st.markdown(f"### {node.get('label', 'Unknown')}")
        st.caption(f"ID: {node.get('id')}")
        
        # 类型标签
        st.markdown(f"**Type:** `{node.get('type', 'Unknown')}`")
        
        # 描述
        st.markdown("#### Description")
        st.info(node.get('desc', 'No description available.'))
        
        # 原始数据
        with st.expander("Raw Data"):
            st.json(node)
            
    else:
        st.info("Select a node from the dropdown above to inspect details.")
        st.markdown("""
        > **Tip:** You can type in the box to search by name or ID.
        """)

    st.divider()
    st.subheader("System Logs")
    log_placeholder = st.empty()
    log_placeholder.code("Ready to weave lore...", language="bash")
