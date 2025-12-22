import json
import os
from src.builder.shadow_builder import ShadowTreeBuilder
from src.llm.processor import LLMProcessor

def main():
    # 1. 路径配置
    input_file = "data/adventure-dosi.json"
    shadow_file = "output/shadow_tree.json"
    output_file = "output/summarized_tree.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(shadow_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # api_key = os.getenv("OPENAI_API_KEY")
    # base_url = os.getenv("OPENAI_BASE_URL") 
    api_key = "sk-dc0a809311a840459553ad6dd9607c3f"
    base_url = "https://api.deepseek.com"

    # 2. 读取数据
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # 5eTools 数据通常在 'data' 键下，或者直接是列表
    adventure_data = raw_data.get("data", []) if isinstance(raw_data, dict) else raw_data

    # 3. 构建影子树
    print("Building Shadow Tree...")
    builder = ShadowTreeBuilder()
    shadow_tree = builder.build(adventure_data)

    # 4. 输出结果
    print(f"Saving Shadow Tree to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shadow_tree, f, indent=2, ensure_ascii=False)

    # # 直接读取已生成的影子树
    # if os.path.exists(shadow_file):
    #     with open(shadow_file, 'r', encoding='utf-8') as f:
    #         shadow_tree = json.load(f)

    # --- 步骤 2: LLM 递归总结 ---
    print("Processing with LLM (This may take a while)...")
    processor = LLMProcessor(
        api_key=api_key,
        base_url=base_url,
        model="deepseek-chat"
    )
    
    summarized_tree = processor.process_tree(shadow_tree)

    # --- 步骤 3: 保存结果 ---
    print(f"Saving result to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summarized_tree, f, indent=2, ensure_ascii=False)

    print("Done! Summarization complete.")

if __name__ == "__main__":
    main()