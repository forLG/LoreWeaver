import hashlib
import json
import re
from pathlib import Path
from typing import Any

from utils.logger import logger
from utils.resolver import InheritanceResolver


# ==========================================
# Utility Functions
# ==========================================
def generate_uuid(type_prefix: str, name: str, source: str) -> str:
    """
    生成确定性的UUID。
    输入组合为：类型前缀 + 名称 + 来源。
    全小写处理以确保ID一致性。
    """
    safe_source = str(source).strip() if source else "unknown"
    safe_name = str(name).strip()
    raw_str = f"{type_prefix}:{safe_name}:{safe_source}".lower()
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

# ==========================================
# Builder Class
# ==========================================
class NodeBuilder:
    """
    主构建器类，解析模组手册中的生物、物品和法术节点。
    """
    def __init__(self, adventure_file_path: str):
        self.nodes = []
        self.stats = {"Monster": 0, "Item": 0, "Spell": 0}
        self.adventure_file_path = adventure_file_path

        # 存储扫描到的引用 (Name, Source)
        self.referenced_entities = {
            "creature": set(),
            "item": set(),
            "spell": set()
        }

        # 初始化 Resolvers
        # 根据要求，属性黑名单位于此处
        common_blacklist = ["otherSources", "variant", "environment", "traitTags", "senseTags", "actionTags", "languageTags", "damageTags", "damageTagsLegendary", "miscTags", "hasToken", "hasFluff", "hasFluffImages"]
        self.monster_resolver = InheritanceResolver(blacklist=common_blacklist)
        self.item_resolver = InheritanceResolver(blacklist=[*common_blacklist, "reqAttuneTags"])
        self.spell_resolver = InheritanceResolver(blacklist=[*common_blacklist, "classes", "classTags"])

    def _load_json(self, path: str) -> dict[str, Any]:
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {path}. skipping.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"JSON decode error in {path}.")
            return {}

    def _init_subsystems(self):
        """加载数据文件并构建索引"""
        logger.info("Initializing subsystems...")

        # 1. Items
        item_files = ["items-base.json", "items.json"]
        item_data = [self._load_json(f) for f in item_files]
        self.item_resolver.build_index(item_data, ["item", "itemGroup", "baseitem"])

        # 2. Monsters
        # 注意：这里需要根据实际文件名修改，或者扫描文件夹
        monster_files = ["bestiary-mm.json", "bestiary-dosi.json"]
        monster_data = [self._load_json(f) for f in monster_files]
        self.monster_resolver.build_index(monster_data, ["monster"])

        # 3. Spells
        spell_files = ["spells-phb.json"] # 假设存在，如不存在需上传
        spell_data = [self._load_json(f) for f in spell_files]
        self.spell_resolver.build_index(spell_data, ["spell"])

    def _scan_references(self):
        """扫描 Adventure JSON 中的 Tag"""
        logger.info(f"Scanning adventure file: {self.adventure_file_path}")
        adventure_data = self._load_json(self.adventure_file_path)

        # 递归扫描函数
        def _scan_recursive(data):
            if isinstance(data, str):
                self._parse_string_tags(data)
            elif isinstance(data, list):
                for item in data:
                    _scan_recursive(item)
            elif isinstance(data, dict):
                for value in data.values():
                    _scan_recursive(value)

        _scan_recursive(adventure_data)

        logger.info(f"Scan complete. Found: "
                    f"{len(self.referenced_entities['creature'])} creatures, "
                    f"{len(self.referenced_entities['item'])} items, "
                    f"{len(self.referenced_entities['spell'])} spells.")

    def _parse_string_tags(self, text: str):
        """解析字符串中的 5e tools 格式标签"""
        # Regex to match {@tag content}
        # groups: (tag_type, content)
        # tag_type examples: creature, item, spell
        pattern = re.compile(r'\{@(creature|item|spell) ([^\}]+)\}')
        matches = pattern.findall(text)

        for tag_type, content in matches:
            # Content format: Name|Source|Text or just Name
            parts = content.split('|')
            name = parts[0].strip()
            source = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

            # 保存到对应的集合中
            self.referenced_entities[tag_type].add((name, source))

    def _process_generic(self, entity_type: str, resolver: InheritanceResolver, label: str, prefix: str):
        """通用的处理逻辑"""
        unique_nodes = {} # 使用Dict去重，key为UUID

        for name, source in self.referenced_entities[entity_type]:
            # 1. 尝试从 Resolver 获取原始数据
            raw_entry = resolver.get_entry_by_tag(name, source)

            if not raw_entry:
                logger.warning(f"Could not resolve {label}: {name} ({source})")
                continue

            # 2. 解析继承关系
            try:
                resolved_entry = resolver.resolve(raw_entry)
            except Exception as e:
                logger.error(f"Error resolving {name}: {e}")
                continue

            # 3. 确定最终的 Name 和 Source (用于生成 UUID)
            final_name = resolved_entry.get("name", name)
            final_source = resolved_entry.get("source", source)

            # 4. 生成 UUID
            node_id = generate_uuid(prefix, final_name, final_source)

            # 5. 构建节点
            if node_id not in unique_nodes:
                node = {
                    "id": node_id,
                    "label": label,
                    "name": final_name,
                    "source": final_source,
                    "attributes": resolved_entry
                }
                unique_nodes[node_id] = node

        # 统计并添加到总列表
        count = len(unique_nodes)
        self.stats[label] = count
        self.nodes.extend(unique_nodes.values())
        logger.info(f"Processed {count} {label} nodes.")

    def _process_monsters(self):
        self._process_generic("creature", self.monster_resolver, "Monster", "monster")

    def _process_items(self):
        self._process_generic("item", self.item_resolver, "Item", "item")

    def _process_spells(self):
        self._process_generic("spell", self.spell_resolver, "Spell", "spell")

    def _export(self):
        """导出结果到 JSON"""
        output = {
            "meta": {
                "stats": self.stats,
                "description": "Phase 1 Skeleton Construction with Attributes Flattened and Inheritance Resolved.",
                "source_file": self.adventure_file_path
            },
            "nodes": self.nodes
        }

        output_filename = "1_vertices_output.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported result to {output_filename}")

    def run(self):
        """执行完整的构建流程"""
        # 1. 初始化子系统
        self._init_subsystems()

        # 1.5 扫描引用
        self._scan_references()

        # 2. 处理各类型数据
        self._process_monsters()
        self._process_items()
        self._process_spells()

        # 3. 导出结果
        self._export()

# ==========================================
# Using Example
# ==========================================
if __name__ == "__main__":
    # 假设你的 adventure json 文件名为 adventure-dosi.json (龙之风暴岛)
    # 如果没有这个文件，脚本会报错，请确保文件存在或修改此处文件名
    # 对于测试，你可以创建一个包含 {@creature Runara} 的 dummy.json

    # 这里演示使用你上传的文件之一作为 adventure file 的场景，或者你需要指定那个包含剧情的 json
    # 既然题目说 "会传入一份 adventure-xxx.json"，这里我们模拟传入参数

    target_adventure = "adventure-dosi.json"

    # 为了演示方便，如果找不到 adventure 文件，我们创建一个假的用于测试
    if not Path(target_adventure).exists():
        logger.warning(f"{target_adventure} not found. Creating a dummy file for demonstration.")
        dummy_data = {
            "data": [
                "The party encounters {@creature Runara} inside the cave.",
                "She gives them a {@item potion of healing}.",
                "The wizard casts {@spell detect magic}."
            ]
        }
        with open(target_adventure, 'w') as f:
            json.dump(dummy_data, f)

    builder = NodeBuilder(adventure_file_path=target_adventure)

    builder.run()
