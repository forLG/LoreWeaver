import streamlit as st
import json
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import sys
import time
import html
import subprocess
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置页面配置
st.set_page_config(page_title="LoreWeaver Holodeck", layout="wide", page_icon="🕸️")

# --- Logger Configuration ---
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "streamlit_app.log"

# Initialize logger explicitly for Streamlit
if "logger_initialized" not in st.session_state:
    try:
        from src.utils.logger import setup_logger
        # Force a specific log file so we can read it back
        setup_logger(exp_name="streamlit_app", log_dir=str(log_dir))
        st.session_state.logger_initialized = True
    except Exception as e:
        print(f"Failed to setup logger: {e}")

import base64

# --- Session State 初始化 ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# --- 辅助函数 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_available_projects():
    """Scans output directory for available projects (json files)."""
    output_dir = project_root / "output"
    if not output_dir.exists():
        return []
    # 获取所有的 .json 文件，排除 final.json（如果它不是项目文件的话，这里假设所有json都是项目）
    files = [f.stem for f in output_dir.glob("*.json") if f.is_file()]
    return sorted(files)

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    /* Create a pseudo-element for the background with blur */
    .stApp::before {{
        content: "";
        position: fixed;
        top: -20px;
        left: -20px;
        width: calc(100% + 40px);
        height: calc(100% + 40px);
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        filter: blur(6px); /* blur */
        z-index: -1;
    }}
    /* Ensure content is readable */
    .stApp {{
        background: rgba(0,0,0,0.2); /* Slight overlay */
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def render_log_viewer(key_prefix="global"):
    """Renders a log viewer component."""
    st.divider()
    st.subheader("🖥️ System Logs")
    
    # Read Content
    log_content = "No logs found."
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                # Get last 50 lines
                lines = f.readlines()
                log_content = "".join(lines[-50:])
        except Exception as e:
            log_content = f"Error reading log file: {e}"
    
    st.code(log_content, language="bash")
    
    # Auto-refresh mechanism could be added here if needed, 
    # but for now we rely on app interactions to trigger re-renders.

# --- 界面逻辑 ---
def landing_page():
    # 尝试加载背景图
    img_path = project_root / "frontend" / "logo.png"
    if img_path.exists():
        set_png_as_page_bg(str(img_path))
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&family=MedievalSharp&display=swap');

        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
        .landing-title {
            text-align: center;
            font-size: 6rem;
            font-family: 'Cinzel Decorative', cursive; /* Artistic Font */
            font-weight: 700;
            color: #FFFFFF;
            text-shadow: 0 0 10px #ff00de, 0 0 20px #000000, 0 0 30px #000000; /* Neon/Magic Glow */
            margin-top: 20vh;
            letter-spacing: 5px;
            animation: glow 3s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #e60073;
            }
            to {
                text-shadow: 0 0 20px #fff, 0 0 30px #ff4da6, 0 0 40px #ff4da6;
            }
        }

        .landing-subtitle {
            text-align: center;
            font-size: 2rem;
            font-family: 'MedievalSharp', cursive;
            color: #EEEEEE;
            text-shadow: 2px 2px 4px #000000;
            margin-bottom: 50px;
            letter-spacing: 2px;
        }
        .stButton > button {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid white;
            border-radius: 10px;
            font-size: 1.5rem;
            font-family: 'Cinzel Decorative', cursive; /* Styled Title Font */
            font-weight: 700;
            padding: 0.5rem 1rem;
            backdrop-filter: blur(5px);
            transition: all 0.3s;
            text-shadow: 2px 2px 4px #000000;
        }
        .stButton > button:hover {
            background-color: rgba(255, 255, 255, 0.5);
            color: #2c3e50;
            transform: scale(1.05);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="landing-title">LoreWeaver</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">The Dungeon Master\'s Intelligent Console</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # Step 1: Click to Enter
        if 'landing_stage' not in st.session_state:
            st.session_state.landing_stage = 'entry'
            
        if st.session_state.landing_stage == 'entry':
            if st.button("⚜️ ENTER HOLODECK ⚜️", use_container_width=True):
                st.session_state.landing_stage = 'menu'
                st.rerun()
        
        # Step 2: Show Options
        else:
            if st.button("🔮 Knowledge Graph Visualization", use_container_width=True):
                st.session_state.page = 'visualization'
                st.rerun()
            
            st.write("") # Spacer
            
            if st.button("📚 Graph RAG (Query System)", use_container_width=True):
                st.session_state.page = 'rag'
                st.rerun()

def rag_page():
    st.title("📚 Graph RAG Interface")
    
    # 初始化 session state
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_prompt' not in st.session_state:
        st.session_state.show_prompt = {}
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 获取可用项目
        projects = get_available_projects()
        
        # 图数据文件选择
        selected_project = st.selectbox(
            "Select Project",
            projects,
            index=0 if projects else None,
            help="Select the project to load"
        )
        
        graph_file = None
        if selected_project:
            graph_file = f"output/{selected_project}.json"
        
        # 搜索参数
        st.subheader("Search Parameters")
        max_hops = st.slider("Max Hops", min_value=1, max_value=3, value=1, 
                            help="Graph traversal radius")
        top_k_nodes = st.slider("Top K Nodes", min_value=1, max_value=10, value=3,
                               help="Number of relevant nodes to retrieve")
        top_k_edges = st.slider("Top K Edges", min_value=1, max_value=10, value=2,
                               help="Number of relevant edges to retrieve")
        
        st.divider()
        
        # 加载图数据
        if st.button("🔄 Load/Reload Graph", use_container_width=True):
            try:
                from src.graphRAG.graph_loader import GraphLoader
                from src.graphRAG.rag_engine import SimpleGraphRAG
                
                full_path = project_root / graph_file
                if not full_path.exists():
                    st.error(f"File not found: {graph_file}")
                else:
                    with st.spinner("Loading graph..."):
                        loader = GraphLoader(str(full_path))
                        graph = loader.load_graph()
                        stats = loader.get_stats()
                        
                        # 初始化 RAG 引擎
                        st.session_state.rag_engine = SimpleGraphRAG(graph)
                        st.session_state.graph_stats = stats
                        st.success(f"✅ Graph loaded! Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
            except Exception as e:
                st.error(f"Error loading graph: {str(e)}")
        
        # 显示图统计信息
        if 'graph_stats' in st.session_state:
            st.metric("Nodes", st.session_state.graph_stats['num_nodes'])
            st.metric("Edges", st.session_state.graph_stats['num_edges'])
        
        st.divider()
        
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.show_prompt = {}
            st.rerun()
        
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = 'landing'
            st.session_state.landing_stage = 'entry'
            st.rerun()
            
        render_log_viewer(key_prefix="rag_sidebar")
    
    # 主界面
    if st.session_state.rag_engine is None:
        st.warning("⚠️ Please load a graph from the sidebar first.")
        st.info("Select a graph file and click 'Load/Reload Graph' to get started.")
        
        # 显示示例问题
        st.markdown("### 💡 Example Questions You Can Ask:")
        st.markdown("""
        - Where are zombies found?
        - What creatures live in the Shadowfell?
        - Tell me about the City of Brass
        - What monsters can I find in dungeons?
        - Describe the relationship between dragons and kobolds
        """)
    else:
        # 聊天界面
        st.markdown("### 💬 Chat with Knowledge Graph")
        
        # 快速示例按钮
        st.markdown("**Quick Examples:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🧟 Zombies", use_container_width=True):
                st.session_state.quick_query = "Where are zombies found?"
        with col2:
            if st.button("🏰 Locations", use_container_width=True):
                st.session_state.quick_query = "What are the main locations?"
        with col3:
            if st.button("🐉 Dragons", use_container_width=True):
                st.session_state.quick_query = "Tell me about dragons"
        with col4:
            if st.button("⚔️ Combat", use_container_width=True):
                st.session_state.quick_query = "What creatures are dangerous in combat?"
        
        st.divider()
        
        # 显示聊天历史
        chat_container = st.container()
        with chat_container:
            for i, entry in enumerate(st.session_state.chat_history):
                # 用户消息
                with st.chat_message("user"):
                    st.markdown(entry['query'])
                
                # AI 回复
                with st.chat_message("assistant"):
                    st.markdown(entry['answer'])
                    
                    # 显示 prompt 按钮
                    col1, col2, col3 = st.columns([1.2, 0.8, 4])
                    with col1:
                        if st.button("🔍 View Prompt", key=f"show_prompt_{i}"):
                            st.session_state.show_prompt[i] = not st.session_state.show_prompt.get(i, False)
                    with col2:
                        if st.button("📋 Copy", key=f"copy_{i}"):
                            st.toast("Answer copied!", icon="✅")
                    
                    # 显示实际的 prompt
                    if st.session_state.show_prompt.get(i, False):
                        with st.expander("📝 Full Prompt Sent to LLM", expanded=True):
                            st.code(entry['prompt'], language="markdown")
                            
                            # 统计信息
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Context Length", f"{len(entry['context'])} chars")
                            with col_b:
                                st.metric("Prompt Length", f"{len(entry['prompt'])} chars")
                            
                            with st.expander("🔎 Retrieved Context (Raw)", expanded=False):
                                st.text(entry['context'])
        
        # 处理快速查询
        if 'quick_query' in st.session_state and st.session_state.quick_query:
            query_to_process = st.session_state.quick_query
            st.session_state.quick_query = None  # 清除状态
            
            with st.spinner("🔍 Searching knowledge graph..."):
                try:
                    # 获取上下文
                    context = st.session_state.rag_engine.retrieve_context(
                        query_to_process, 
                        max_hops=max_hops, 
                        top_k=top_k_nodes, 
                        top_k_edges=top_k_edges
                    )
                    
                    # 生成答案
                    answer = st.session_state.rag_engine.answer_query(query_to_process, use_llm=True)
                    
                    # 构建完整的 prompt（用于显示）
                    full_prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge graph context.

Context:
{context}

User Query: {query_to_process}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""
                    
                    # 保存到历史记录
                    st.session_state.chat_history.append({
                        'query': query_to_process,
                        'answer': answer,
                        'context': context,
                        'prompt': full_prompt
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # 输入框
        query_input = st.chat_input("💭 Ask a question about the knowledge graph...")
        
        if query_input:
            with st.spinner("🔍 Searching knowledge graph..."):
                try:
                    # 获取上下文
                    context = st.session_state.rag_engine.retrieve_context(
                        query_input, 
                        max_hops=max_hops, 
                        top_k=top_k_nodes, 
                        top_k_edges=top_k_edges
                    )
                    
                    # 生成答案
                    answer = st.session_state.rag_engine.answer_query(query_input, use_llm=True)
                    
                    # 构建完整的 prompt（用于显示）
                    full_prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge graph context.

Context:
{context}

User Query: {query_input}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""
                    
                    # 保存到历史记录
                    st.session_state.chat_history.append({
                        'query': query_input,
                        'answer': answer,
                        'context': context,
                        'prompt': full_prompt
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# 路由控制
if st.session_state.page == 'landing':
    landing_page()
    st.stop() # 停止执行后续代码
elif st.session_state.page == 'rag':
    rag_page()
    st.stop()

# 下面是原有的主程序代码（visualization 页面）
# Only add a "Back to Home" button in sidebar for navigation
if st.session_state.page == 'visualization':
    with st.sidebar:
        if st.button("🏠 Back to Home"):
            st.session_state.page = 'landing'
            st.session_state.landing_stage = 'entry'
            st.rerun()

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

st.title("LoreWeaver: Dungeon Master's Console")

# --- 侧边栏：控制面板 ---
with st.sidebar:
    st.header("Project Selection")
    
    # 获取可用项目
    projects = get_available_projects()
    current_project = st.selectbox(
        "Select Adventure", 
        projects, 
        index=0 if projects else None,
        help="Select which adventure's output to visualize."
    )
    st.session_state.current_project = current_project
    
    st.divider()
    
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
            # 模型选择通过环境变量 LLM_MODEL 传递；main.py 从环境中读取模型配置，而不是使用 --model 参数。
            
            
            
            
            
            
            
            
            
            
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
        ["Entity Graph", "Location Graph"],
        index=0
    )

# --- 主界面：图谱可视化 ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Interactive View: {graph_source}")
    
    def get_graph_path(source_name, project_name):
        if not project_name:
            return None
            
        base_output = project_root / "output"
        base_cache = project_root / "cache" / project_name
        
        if source_name == "Entity Graph":
            return base_output / f"{project_name}.json"
        elif source_name == "Location Graph":
            return base_cache / "location_graph.json"
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
    target_path = get_graph_path(graph_source, st.session_state.get("current_project"))
    graph_data = load_graph_data(target_path)
    
    # 2. Focus State Logic
    if 'focus_target' not in st.session_state:
        st.session_state.focus_target = None
        
    focus_target = st.session_state.focus_target
    
    # Header area for graph view (Reset button)
    if focus_target:
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            st.info(f"Focusing on: **{focus_target}**")
        with hdr_col2:
            if st.button("❌ Clear Focus"):
                st.session_state.focus_target = None
                st.rerun()

    # 3. 提取类型并显示过滤器
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
    
    def visualize_graph(data, use_physics, filter_types=None, focus_id=None):
        net = Network(height="600px", width="100%", bgcolor="#f0f2f6", font_color="#333333")
        
        if data:
            try:
                node_count = 0
                edge_count = 0
                
                if "nodes" in data and "edges" in data:
                    # 标准图结构
                    existing_nodes = set()
                    highlighted_ids = set() # 记录高亮的节点ID
                    
                    # Pre-calculate neighbor set if focus is active
                    focus_group = set()
                    if focus_id:
                        focus_group.add(focus_id)
                        for edge in data['edges']:
                            if edge['source'] == focus_id:
                                focus_group.add(edge['target'])
                            elif edge['target'] == focus_id:
                                focus_group.add(edge['source'])

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
                        
                        # Logic: Focus Overrides Filter
                        if focus_id:
                            if node['id'] not in focus_group:
                                is_highlighted = False
                        else:
                            # Normal type filtering
                            if filter_types is not None and normalized_type not in filter_types:
                                is_highlighted = False

                        node_id = node['id']
                        # Strict New JSON: valid label > id (key). No 'name' fallback.
                        label = node.get('label') or node_id
                        title = node.get('desc', '') or ""
                        
                        # 计算节点大小 (基础大小 25 + 度数 * 3) - 调大基础大小以容纳文字
                        # 限制最大大小以防过大
                        node_degree = degrees.get(node_id, 0)
                        size = 25 + min(node_degree * 3, 60)  
                        
                        # 默认颜色逻辑 (User Custom Pastel Palette)
                        color = "#34CAF7" # Default
                        
                        if 'Location' in normalized_type: 
                            color = "#f1b8f1" # Light Pink
                        elif 'Monster' in normalized_type: 
                            color = "#d9b8f1" # Pastel Purple
                        elif 'Npc' in normalized_type: 
                            color = '#f1ccb8' # Pastel Green
                        elif 'Player' in normalized_type: 
                            color = '#f1f1b8' # Pastel Purple
                        elif 'Item' in normalized_type: 
                            color = '#b8f1ed' # Pastel Orange
                        elif 'Event' in normalized_type:
                            color = '#b8f1cc' # Pastel Yellow
                        
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
                            
                            # Strict New JSON format (relationship, desc, weight)
                            relation_type = edge.get('relationship', '')
                            edge_desc = edge.get('desc', '')
                            weight = edge.get('weight')
                            
                            # Build edge tooltip
                            edge_title = relation_type
                            if edge_desc:
                                edge_title += f"\n\n{edge_desc}"
                            if weight:
                                edge_title += f"\n(Weight: {weight})"
                            
                            # Use weight for edge width if available
                            edge_width = 1
                            if weight:
                                try:
                                    # Scale weight to reasonable width (e.g. 1-5)
                                    edge_width = max(1, min(float(weight), 10))
                                except:
                                    pass
                                
                            net.add_edge(source, target, title=edge_title, label=relation_type, color=edge_color, width=edge_width)
                            edge_count += 1
                        except Exception as e:
                            print(f"Skipping edge {source} -> {target}: {e}")
                            continue
                
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
                  "damping": 1.2,
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

    html_path = visualize_graph(graph_data, physics, selected_types, focus_target)

    
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
        
        st.divider()
        if st.button("🎯 Highlight in Graph", type="primary", use_container_width=True):
            st.session_state.focus_target = node['id']
            st.rerun()
            
    else:
        st.info("Select a node from the dropdown above to inspect details.")
        st.markdown("""
        > **Tip:** You can type in the box to search by name or ID.
        """)

    # 使用新的 Log Viewer
    render_log_viewer(key_prefix="viz_sidebar")
