import sys
import os
import time
import base64
import html
import subprocess
import json  # Added json import
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

st.set_page_config(page_title="LoreWeaver Holodeck", layout="wide", page_icon="🕸️")

# --- Logger Configuration ---
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "streamlit_app.log"

# Initialize logger explicitly for Streamlit
if "logger_initialized" not in st.session_state:
    try:
        from src.utils.logger import setup_logger
        # Force a specific log file to enable reading it back in the UI
        setup_logger(exp_name="streamlit_app", log_dir=str(log_dir))
        st.session_state.logger_initialized = True
    except Exception as e:
        print(f"Failed to setup logger: {e}")

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# --- Helper Functions ---

def get_base64_of_bin_file(bin_file):
    """Reads a binary file and returns its base64 encoded string.

    Args:
        bin_file (str): Path to the binary file.

    Returns:
        str: Base64 encoded string of the file content.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_available_projects():
    """Scans output directory for available projects (json files).

    Returns:
        list: Sorted list of project names (stems of json files).
    """
    output_dir = project_root / "output"
    if not output_dir.exists():
        return []
    # Get all .json files
    files = [f.stem for f in output_dir.glob("*.json") if f.is_file()]
    return sorted(files)

def set_png_as_page_bg(png_file):
    """Sets a PNG file as the page background with a blur effect.

    Args:
        png_file (str): Path to the PNG file.
    """
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
    """Renders a log viewer component.

    Args:
        key_prefix (str): Prefix for unique keys in Streamlit components.
    """
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
    
    # Auto-refresh mechanism could be added here if needed.

# --- Interface Logic ---

def landing_page():
    """Renders the landing page of the application."""
    # Attempt to load background image
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
    """Renders the Graph RAG (Retrieval-Augmented Generation) interface."""
    st.title("📚 Graph RAG Interface")
    
    # Initialize session state
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_prompt' not in st.session_state:
        st.session_state.show_prompt = {}
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get available projects
        projects = get_available_projects()
        
        # Select Graph Project
        selected_project = st.selectbox(
            "Select Project",
            projects,
            index=0 if projects else None,
            help="Select the project to load"
        )
        
        graph_file = None
        if selected_project:
            graph_file = f"output/{selected_project}.json"
        
        # Search Parameters
        st.subheader("Search Parameters")
        max_hops = st.slider("Max Hops", min_value=1, max_value=3, value=1, 
                            help="Graph traversal radius")
        top_k_nodes = st.slider("Top K Nodes", min_value=1, max_value=10, value=3,
                               help="Number of relevant nodes to retrieve")
        top_k_edges = st.slider("Top K Edges", min_value=1, max_value=10, value=2,
                               help="Number of relevant edges to retrieve")
        
        st.divider()
        
        # Load Graph Data
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
        
        # Display Graph Statistics
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
    
    # Main Interface
    if st.session_state.rag_engine is None:
        st.warning("⚠️ Please load a graph from the sidebar first.")
        st.info("Select a graph file and click 'Load/Reload Graph' to get started.")
        
        # Display Example Questions
        st.markdown("### 💡 Example Questions You Can Ask:")
        st.markdown("""
        - Where are zombies found?
        - What creatures live in the Shadowfell?
        - Tell me about the City of Brass
        - What monsters can I find in dungeons?
        - Describe the relationship between dragons and kobolds
        """)
    else:
        # Chat Interface
        st.markdown("### 💬 Chat with Knowledge Graph")
        
        # Quick Example Buttons
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
        
        # Display Chat History
        chat_container = st.container()
        with chat_container:
            for i, entry in enumerate(st.session_state.chat_history):
                # User Message
                with st.chat_message("user"):
                    st.markdown(entry['query'])
                
                # AI Response
                with st.chat_message("assistant"):
                    st.markdown(entry['answer'])
                    
                    # Show Prompt Button
                    col1, col2, col3 = st.columns([1.2, 0.8, 4])
                    with col1:
                        if st.button("🔍 View Prompt", key=f"show_prompt_{i}"):
                            st.session_state.show_prompt[i] = not st.session_state.show_prompt.get(i, False)
                    with col2:
                        if st.button("📋 Copy", key=f"copy_{i}"):
                            st.toast("Answer copied!", icon="✅")
                    
                    # Show Actual Prompt
                    if st.session_state.show_prompt.get(i, False):
                        with st.expander("📝 Full Prompt Sent to LLM", expanded=True):
                            st.code(entry['prompt'], language="markdown")
                            
                            # statistics
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
            st.session_state.quick_query = None  # Clear state
            
            with st.spinner("🔍 Searching knowledge graph..."):
                try:
                    # Retrieve Context
                    context = st.session_state.rag_engine.retrieve_context(
                        query_to_process, 
                        max_hops=max_hops, 
                        top_k=top_k_nodes, 
                        top_k_edges=top_k_edges
                    )
                    
                    # Generate Answer
                    answer = st.session_state.rag_engine.answer_query(query_to_process, use_llm=True)
                    
                    # Construct Prompt (for display)
                    full_prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge graph context.

Context:
{context}

User Query: {query_to_process}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""
                    
                    # Save to History
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
        
        # Input Box
        query_input = st.chat_input("💭 Ask a question about the knowledge graph...")
        
        if query_input:
            with st.spinner("🔍 Searching knowledge graph..."):
                try:
                    # Retrieve Context
                    context = st.session_state.rag_engine.retrieve_context(
                        query_input, 
                        max_hops=max_hops, 
                        top_k=top_k_nodes, 
                        top_k_edges=top_k_edges
                    )
                    
                    # Generate Answer
                    answer = st.session_state.rag_engine.answer_query(query_input, use_llm=True)
                    
                    # Construct Prompt (for display)
                    full_prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge graph context.

Context:
{context}

User Query: {query_input}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""
                    
                    # Save to History
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

# Routing Control
if st.session_state.page == 'landing':
    landing_page()
    st.stop()
elif st.session_state.page == 'rag':
    rag_page()
    st.stop()

# Visualization Page Logic
if st.session_state.page == 'visualization':
    with st.sidebar:
        if st.button("🏠 Back to Home"):
            st.session_state.page = 'landing'
            st.session_state.landing_stage = 'entry'
            st.rerun()

# --- CSS Styling ---
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

# --- Sidebar: Control Panel ---
with st.sidebar:
    st.header("Project Selection")
    
    # Get available projects
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
    
    # Model Selection
    model_choice = st.selectbox(
        "Model", 
        ["deepseek-chat", "qwen3-8b"], 
        index=0,
        help="Select the LLM model to use for processing."
    )
    
    # Stage Selection
    stages = st.multiselect(
        "Stages", 
        ["shadow", "spatial", "section-map", "entity"], 
        default=["entity"],
        help="Select pipeline stages to run."
    )
    
    # Concurrency Control
    concurrency = st.slider("Concurrency", min_value=1, max_value=50, value=5)
    
    st.divider()
    
    # File Upload
    uploaded_file = st.file_uploader("Upload Adventure JSON", type=['json'])
    input_file_path = None
    
    if uploaded_file is not None:
        # Save uploaded file to data directory
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
            
            # Build command
            cmd = [sys.executable, "-m", "src.main"]
            
            # Enforce correct stage order: shadow -> spatial -> section-map -> entity
            ORDERED_STAGES = ["shadow", "spatial", "section-map", "entity"]
            sorted_stages = [s for s in ORDERED_STAGES if s in stages]
            
            for stage in sorted_stages:
                cmd.extend(["--stage", stage])
            
            # Specify input if file uploaded
            if input_file_path:
                cmd.extend(["--input", str(input_file_path)])
            
            # Set env vars for compatibility
            env = os.environ.copy()
            env["LLM_MODEL"] = model_choice
            env["LLM_MAX_CONCURRENT"] = str(concurrency)
            
            # Explicitly pass CLI arguments
            cmd.extend(["--max-concurrent", str(concurrency)])
            
            st.code(" ".join(cmd), language="bash")
            
            log_container = st.empty()
            output_lines = []
            
            try:
                # Execute using subprocess.Popen
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
                
                # Real-time output reading
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
                    # Clear cache to reload latest graph
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
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

# --- Main Interface: Graph Visualization ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Interactive View: {graph_source}")
    
    def get_graph_path(source_name, project_name):
        """Resolves the file path for the requested graph."""
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
        """Loads graph data from a JSON file.

        Args:
            file_path (Path): Path to the JSON file.

        Returns:
            dict: Parsed JSON data or None if failed.
        """
        if file_path and file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
        return None

    # Load Data
    target_path = get_graph_path(graph_source, st.session_state.get("current_project"))
    graph_data = load_graph_data(target_path)
    
    # Focus State Logic
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

    # Extract types and show filters
    all_types = []
    if graph_data and "nodes" in graph_data:
        # Normalize case
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
        """Generates a PyVis network graph.

        Args:
            data (dict): Graph data containing nodes and edges.
            use_physics (bool): Whether to enable physics simulation.
            filter_types (list, optional): List of node types to display.
            focus_id (str, optional): Node ID to focus on.

        Returns:
            Path: Path to the generated HTML file.
        """
        net = Network(height="600px", width="100%", bgcolor="#f0f2f6", font_color="#333333")
        
        if data:
            try:
                node_count = 0
                edge_count = 0
                
                if "nodes" in data and "edges" in data:
                    existing_nodes = set()
                    highlighted_ids = set() 
                    
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
                        
                        # Calculate node size based on degree, 
                        # limiting max size to prevent overly large nodes
                        node_degree = degrees.get(node_id, 0)
                        size = 25 + min(node_degree * 3, 60)  
                        
                        # User Custom Pastel Palette
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
                        
                        if not is_highlighted:
                            color = 'rgba(50, 50, 50, 0.1)' # Dimmed
                        else:
                            highlighted_ids.add(node_id)
                        
                        desc = str(node.get('desc', ''))
                        if not desc or desc.lower() == 'no description available.':
                            desc = ''
                        
                        safe_label = html.escape(label)
                        safe_type = html.escape(normalized_type)
                        
                        tooltip_text = f"【{label}】\n[{normalized_type}]"
                        
                        if desc:
                            tooltip_text += f"\n\n{desc}"

                        # Truncate labels based on size to prevent overlap
                        max_chars = max(4, int(size / 3.5))
                        display_label = label
                        if len(label) > max_chars:
                            display_label = label[:max_chars] + ".."

                        # Font style
                        font_style = {
                            'color': '#2c3e50', 
                            'size': 14, 
                            'face': 'arial',
                            'strokeWidth': 3,
                            'strokeColor': '#ffffff',
                            'bold': True,
                            'vadjust': 0
                        }

                        net.add_node(
                            node_id, 
                            label=display_label, 
                            title=tooltip_text, 
                            color=color, 
                            size=size,
                            shape='circle',
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
            # Enable physics with parameters to reduce oscillation
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
        
        # Save to temp file using absolute path
        tmp_path = project_root / "frontend" / "temp_graph.html"
        net.save_graph(str(tmp_path))
        return tmp_path

    html_path = visualize_graph(graph_data, physics, selected_types, focus_target)

    if html_path and html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
            # --- Inject custom CSS for Tooltip ---
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
                    white-space: pre-wrap !important;
                    z-index: 99999 !important;
                }
            </style>
            """
            source_code = source_code.replace("</head>", f"{custom_css}</head>")
            
        components.html(source_code, height=610)

# --- Right Sidebar: Details ---
with col2:
    st.subheader("Entity Inspector")
    
    # Create lookup dict for nodes (Multi-index: ID and Label)
    node_lookup = {}
    if graph_data and "nodes" in graph_data:
        for node in graph_data['nodes']:
            node_lookup[node['id']] = node
            if 'label' in node:
                node_lookup[node['label']] = node
            if 'name' in node:
                node_lookup[node['name']] = node

    # Get all node names for search suggestions
    search_options = sorted(list(node_lookup.keys()))

    selected_entity_id = st.selectbox(
        "Select or Search Entity",
        options=[""] + search_options,
        format_func=lambda x: x if x else "Type to search...",
        help="Select a node to view details."
    )

    if selected_entity_id and selected_entity_id in node_lookup:
        node = node_lookup[selected_entity_id]
        
        # Display details card
        st.markdown(f"### {node.get('label', 'Unknown')}")
        st.caption(f"ID: {node.get('id')}")
        
        st.markdown(f"**Type:** `{node.get('type', 'Unknown')}`")
        
        st.markdown("#### Description")
        st.info(node.get('desc', 'No description available.'))
        
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

    # Use Log Viewer
    render_log_viewer(key_prefix="viz_sidebar")
