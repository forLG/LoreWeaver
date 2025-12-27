import json
import os
from collections import Counter
from src.builder.shadow_builder import ShadowTreeBuilder
# from src.llm.summary_processor import SummaryProcessor
from src.llm.spatial_processor import SpatialTopologyProcessor, SectionLocationMapper
from src.llm.entity_processor import EntityProcessor

def main():
    # 1. 路径配置
    input_file = "data/adventure-dosi.json"
    shadow_file = "output/shadow_tree.json"
    intermediate_file = "output/shadow_tree_with_spatial_summary.json"
    location_graph_file = "output/location_graph.json"
    section_location_map_file = "output/section_location_map.json"
    entity_graph_file = "output/entity_graph.json"

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

    skip_summary = False
    if os.path.exists(intermediate_file):
        print(f"Found intermediate file {intermediate_file}, skipping summarization step...")
        with open(intermediate_file, 'r', encoding='utf-8') as f:
            shadow_tree = json.load(f)
        skip_summary = True

    # # --- 步骤 3: 提取空间拓扑图谱 (核心任务) ---
    # print("Extracting Spatial Topology Graph...")
    # spatial_processor = SpatialTopologyProcessor(
    #     api_key=api_key,
    #     base_url=base_url,
    #     model="deepseek-chat"
    # )
    
    # location_graph = spatial_processor.process(shadow_tree, skip_summary=skip_summary)

    # if not skip_summary:
    #     print(f"Saving intermediate summaries to {intermediate_file}...")
    #     with open(intermediate_file, 'w', encoding='utf-8') as f:
    #         json.dump(shadow_tree, f, indent=2, ensure_ascii=False)

    # # --- 步骤 4: 保存图谱结果 ---
    # print(f"Saving Location Graph to {location_graph_file}...")
    # with open(location_graph_file, 'w', encoding='utf-8') as f:
    #     json.dump(location_graph, f, indent=2, ensure_ascii=False)

    # print(f"Done! Graph contains {len(location_graph['nodes'])} nodes and {len(location_graph['edges'])} edges.")

    # 提取章节到地点的映射
    if os.path.exists(section_location_map_file):
        with open(location_graph_file, 'r', encoding='utf-8') as f:
            location_graph = json.load(f)

    print("Mapping Sections to Locations...")
    mapper = SectionLocationMapper(
        api_key=api_key,
        base_url=base_url,
        model="deepseek-chat"
    )
    
    section_map = mapper.process(shadow_tree, location_graph)
    
    print(f"Saving Section-Location Map to {section_location_map_file}...")
    with open(section_location_map_file, 'w', encoding='utf-8') as f:
        json.dump(section_map, f, indent=2, ensure_ascii=False)

    if os.path.exists(section_location_map_file):
        with open(section_location_map_file, 'r', encoding='utf-8') as f:
            section_map = json.load(f)

    # --- 步骤 6: 实体实例化与关系挖掘 ---
    print("Extracting Entities and Relations...")
    entity_processor = EntityProcessor(
        api_key=api_key,
        base_url=base_url,
        model="deepseek-chat"
    )
    
    entity_graph = entity_processor.process(shadow_tree, section_map)
    
    print(f"Saving Entity Graph to {entity_graph_file}...")
    with open(entity_graph_file, 'w', encoding='utf-8') as f:
        json.dump(entity_graph, f, indent=2, ensure_ascii=False)
        
    print(f"Entity Graph contains {len(entity_graph['nodes'])} nodes and {len(entity_graph['edges'])} edges.")

    if os.path.exists(entity_graph_file):
        with open(entity_graph_file, 'r', encoding='utf-8') as f:
            entity_graph = json.load(f)

    type_counts = Counter()
    for node in entity_graph["nodes"]:
        node_type = node.get("type", "Unknown").title()
        type_counts[node_type] += 1
    
    for node_type, count in type_counts.most_common():
        print(f"{node_type}: {count}")

if __name__ == "__main__":
    main()