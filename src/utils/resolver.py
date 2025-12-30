import copy
import re
from typing import Any

from utils.logger import logger


class InheritanceResolver:
    def __init__(self, blacklist: list[str] | None = None):
        self.index: dict[str, dict[str, Any]] = {}
        self.blacklist = set(blacklist) if blacklist else set()
        # source_map key 存储全小写 name，value 存储原始 source 字符串
        self.source_map: dict[str, str] = {}

    def build_index(self, data_sources: list[dict[str, Any]], type_keys: list[str]):
        count = 0
        for source_data in data_sources:
            for key in type_keys:
                if key in source_data and isinstance(source_data[key], list):
                    for entry in source_data[key]:
                        name = entry.get("name")
                        source = entry.get("source")

                        if name and source:
                            idx_key = self._get_key(name, source)
                            self.index[idx_key] = entry

                            # 建立 Name -> Default Source 的映射
                            # 统一使用 strip() 和 lower() 作为 Key
                            name_key = str(name).strip().lower()
                            if name_key not in self.source_map:
                                self.source_map[name_key] = source
                            count += 1
        logger.info(f"Built index with {count} entities from keys: {type_keys}")

    def _get_key(self, name: str, source: str) -> str:
        """
        生成唯一索引键。
        强制转换为字符串，去除首尾空格，并转为小写。
        """
        n = str(name).strip().lower()
        s = str(source).strip().lower()
        return f"{n}::{s}"

    def get_entry_by_tag(self, name: str, source: str | None) -> dict[str, Any] | None:
        """
        根据标签信息查找原始条目。
        不区分大小写。
        """
        # 1. 统一处理输入的 name
        safe_name = str(name).strip()
        name_lookup_key = safe_name.lower()

        # 2. 如果没有提供 source，尝试从 source_map 查找默认 source
        if not source:
            source = self.source_map.get(name_lookup_key)
            if not source:
                # TODO: 找不到默认来源，无法精确定位
                return None

        # 3. 生成 Key (内部会自动 lower) 并查找
        # 这里的 source 可能是从 Tag 传入的 (可能大写)，也可能是 source_map 拿到的 (可能大写)
        # _get_key 会再次 lower() 确保匹配 index 中的 Key
        key = self._get_key(safe_name, source)
        return self.index.get(key)

    def resolve(self, entry: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        """递归解析继承"""
        if depth > 10:
            logger.warning(f"Recursion depth limit reached for {entry.get('name')}")
            return entry

        if "_copy" not in entry:
            return self._finalize_entry(entry)

        copy_meta = entry["_copy"]
        parent_name = copy_meta.get("name")
        parent_source = copy_meta.get("source")

        # 获取父级 Key
        parent_key = self._get_key(parent_name, parent_source)

        # 如果直接找不到，尝试通过 source_map 模糊查找
        if parent_key not in self.index:
            fallback_source = self.source_map.get(str(parent_name).strip().lower())
            if fallback_source:
                parent_key = self._get_key(parent_name, fallback_source)

            if parent_key not in self.index:
                logger.warning(f"Parent entity not found: {parent_name} ({parent_source})")
                cleaned = entry.copy()
                cleaned.pop("_copy", None)
                return self._finalize_entry(cleaned)

        raw_parent = self.index[parent_key]
        resolved_parent = self.resolve(raw_parent, depth + 1)
        merged_entry = copy.deepcopy(resolved_parent)

        if "_mod" in copy_meta:
            self._apply_mods(merged_entry, copy_meta["_mod"])

        for key, value in entry.items():
            if key in ["_copy", "_mod", "_preserve"]:
                continue
            merged_entry[key] = value

        return self._finalize_entry(merged_entry)

    def _finalize_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """清理条目"""
        for key in list(entry.keys()):
            if key in self.blacklist or key.startswith("_"):
                del entry[key]
        return entry

    def _apply_mods(self, entity: dict[str, Any], mods: dict[str, Any]):
        for target_key, mod_actions in mods.items():
            if target_key == "*":
                self._apply_text_mod(entity, mod_actions)
                continue
            if target_key not in entity:
                continue

            if isinstance(entity[target_key], list):
                actions = mod_actions if isinstance(mod_actions, list) else [mod_actions]
                for action in actions:
                    self._apply_list_mod(entity[target_key], action)

    def _apply_text_mod(self, entity: Any, mod: dict[str, str]):
        mode = mod.get("mode")
        if mode != "replaceTxt":
            return
        replace_pattern = mod.get("replace")
        with_str = mod.get("with")
        flags_str = mod.get("flags", "")
        flags = re.IGNORECASE if "i" in flags_str else 0
        try:
            pattern = re.compile(replace_pattern, flags)

            def _rec(obj):
                if isinstance(obj, str):
                    return pattern.sub(with_str, obj)
                elif isinstance(obj, list):
                    return [_rec(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: _rec(v) for k, v in obj.items()}
                return obj
            if isinstance(entity, dict):
                for k, v in entity.items():
                    entity[k] = _rec(v)
        except Exception:
            pass

    def _apply_list_mod(self, target_list: list[Any], mod: dict[str, Any]):
        mode = mod.get("mode")
        if mode == "removeArr":
            names = mod.get("names")
            if isinstance(names, str):
                names = [names]
            for i in range(len(target_list) - 1, -1, -1):
                item = target_list[i]
                n = item.get("name") if isinstance(item, dict) else item
                if n in names:
                    target_list.pop(i)
        elif mode == "replaceArr":
            target = mod.get("replace")
            new_item = mod.get("items")
            for i, item in enumerate(target_list):
                n = item.get("name") if isinstance(item, dict) else item
                if n == target:
                    target_list[i] = new_item
                    break
        elif mode == "appendArr":
            target_list.extend(mod.get("items") if isinstance(mod.get("items"), list) else [mod.get("items")])

if __name__ == "__main__":
    import json

    black_list = ["otherSources", "variant", "environment", "traitTags", "senseTags", "actionTags", "languageTags", "damageTags", "damageTagsLegendary", "miscTags"]

    data_sources = []
    for path in ['../data/bestiary/bestiary-dosi.json', '../data/bestiary/bestiary-mm.json']:
        with open(path, encoding='utf-8') as f:
            data_sources.append(json.load(f))

    resolver = InheritanceResolver(blacklist=black_list)

    # 1. 建立索引
    resolver.build_index(data_sources, type_keys=["monster"])

    # 2. 找到你要解析的 Runara 数据 (模拟)
    runara_entry = {
			"name": "Runara",
			"isNpc": True,
			"isNamedCreature": True,
			"source": "DoSI",
			"page": 40,
			"_copy": {
				"name": "Adult Bronze Dragon",
				"source": "MM",
				"_mod": {
					"*": {
						"mode": "replaceTxt",
						"replace": "the dragon",
						"with": "Runara",
						"flags": "i"
					},
					"action": [
						{
							"mode": "removeArr",
							"names": "Tail"
						},
						{
							"mode": "replaceArr",
							"replace": "Change Shape",
							"items": {
								"name": "Change Shape",
								"type": "entries",
								"entries": [
									"Runara magically transforms into a Humanoid or Beast that is Medium or Small, while retaining her game statistics (other than her size). This transformation ends if Runara is reduced to 0 hit points or uses a bonus action to end it."
								]
							}
						}
					]
				}
			},
			"cr": "13",
			"legendary": None,
			"hasToken": True,
			"hasFluff": True,
			"hasFluffImages": True
		}

    # 3. 解析
    resolved_runara = resolver.resolve(runara_entry)

    # 输出结果将包含合并了 Adult Bronze Dragon 的数据，应用了 Runara 的修改，并移除了黑名单标签。
    print(json.dumps(resolved_runara, indent=2))
