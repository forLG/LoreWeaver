import json
import os
from src.builder.shadow_builder import ShadowTreeBuilder
# from src.llm.summary_processor import SummaryProcessor
from src.llm.spatial_processor import SpatialTopologyProcessor

def main():
    # 1. 路径配置
    input_file = "data/adventure-dosi.json"
    shadow_file = "output/test_llm.json"
    # summary_output_file = "output/summary_tree.json"
    graph_output_file = "output/location_graph.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(shadow_file), exist_ok=True)
    # os.makedirs(os.path.dirname(summary_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(graph_output_file), exist_ok=True)

    # api_key = os.getenv("OPENAI_API_KEY")
    # base_url = os.getenv("OPENAI_BASE_URL") 
    api_key = "sk-dc0a809311a840459553ad6dd9607c3f"
    base_url = "https://api.deepseek.com"

    # # 2. 读取数据
    # print(f"Loading data from {input_file}...")
    # try:
    #     with open(input_file, 'r', encoding='utf-8') as f:
    #         raw_data = json.load(f)
    # except FileNotFoundError:
    #     print("Error: Input file not found.")
    #     return

    # # 5eTools 数据通常在 'data' 键下，或者直接是列表
    # adventure_data = raw_data.get("data", []) if isinstance(raw_data, dict) else raw_data

    # # 3. 构建影子树
    # print("Building Shadow Tree...")
    # builder = ShadowTreeBuilder()
    # shadow_tree = builder.build(adventure_data)

    # # 保存影子树（可选，用于调试）
    # print(f"Saving Shadow Tree to {shadow_file}...")
    # with open(shadow_file, 'w', encoding='utf-8') as f:
    #     json.dump(shadow_tree, f, indent=2, ensure_ascii=False)

    # 直接读取已生成的影子树
    if os.path.exists(shadow_file):
        with open(shadow_file, 'r', encoding='utf-8') as f:
            shadow_tree = json.load(f)

    # --- 步骤 2: LLM 递归总结 (可选，如果你还需要普通摘要) ---
    # print("Processing Summaries with LLM...")
    # summary_processor = LLMProcessor(
    #     api_key=api_key,
    #     base_url=base_url,
    #     model="deepseek-chat"
    # )
    # summarized_tree = summary_processor.process_tree(shadow_tree)
    # with open(summary_output_file, 'w', encoding='utf-8') as f:
    #     json.dump(summarized_tree, f, indent=2, ensure_ascii=False)

    # --- 步骤 3: 提取空间拓扑图谱 (核心任务) ---
    print("Extracting Spatial Topology Graph...")
    spatial_processor = SpatialTopologyProcessor(
        api_key=api_key,
        base_url=base_url,
        model="deepseek-chat"
    )
    
    # 注意：这里传入的是 shadow_tree，processor 会在内部递归处理
    location_graph = spatial_processor.process(shadow_tree)

    intermediate_file = "output/shadow_tree_with_spatial_summary.json"
    print(f"Saving intermediate summaries to {intermediate_file}...")
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        json.dump(shadow_tree, f, indent=2, ensure_ascii=False)

    # --- 步骤 4: 保存图谱结果 ---
    print(f"Saving Location Graph to {graph_output_file}...")
    with open(graph_output_file, 'w', encoding='utf-8') as f:
        json.dump(location_graph, f, indent=2, ensure_ascii=False)

    print(f"Done! Graph contains {len(location_graph['nodes'])} nodes and {len(location_graph['edges'])} edges.")

if __name__ == "__main__":
    main()