import json
import os
from src.builder.shadow_builder import ShadowTreeBuilder

def main():
    # 1. 路径配置
    input_file = "data/adventure-dosi.json"
    output_file = "output/shadow_tree.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
    print(f"Saving result to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shadow_tree, f, indent=2, ensure_ascii=False)

    print("Done! Shadow Tree generated successfully.")

if __name__ == "__main__":
    main()