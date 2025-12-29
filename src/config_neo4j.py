"""
Neo4j Configuration for LoreWeaver

配置说明：
1. 修改 NEO4J_* 配置以匹配你的 Neo4j 实例
2. 默认端口：7687 (Bolt), 7474 (HTTP)
3. Windows 本地安装：https://neo4j.com/download/
"""
import os
from pathlib import Path

# 项目根目录（config_neo4j.py 在 src/ 目录下）
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "qwen3"

# Neo4j 连接配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # 请修改为实际密码
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# 输入文件路径
LOCATION_GRAPH_FILE = OUTPUT_DIR / "location_graph.json"
ENTITY_GRAPH_FILE = OUTPUT_DIR / "entity_graph.json"

# 可视化输出路径
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# 批处理大小（根据内存调整）
BATCH_SIZE = 500

# 可视化配置
VISUALIZATION_CONFIG = {
    'width': '100%',
    'height': '900px',
    'bgcolor': '#1a1a1a',
    'font_color': 'white',
    'physics': True
}
