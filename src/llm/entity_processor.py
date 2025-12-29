import json
import asyncio
from typing import List, Dict, Any, Set
from openai import AsyncOpenAI
from utils.logger import logger
from llm.prompt_factory import PromptFactory

class EntityProcessor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "deepseek-chat", max_concurrent: int = 100):
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def process(self, shadow_tree: List[Dict], section_map: Dict[str, List[str]]) -> Dict:
        """
        同步入口
        """
        return asyncio.run(self._process_async(shadow_tree, section_map))

    async def _process_async(self, shadow_tree: List[Dict], section_map: Dict[str, List[str]]) -> Dict:
        # 1. 收集所有需要处理的 Section
        sections = self._collect_sections(shadow_tree)
        logger.info(f"Starting Entity Extraction for {len(sections)} sections...")

        tasks = []
        for section in sections:
            # 获取该章节对应的地点列表
            loc_ids = section_map.get(section["id"], [])
            
            # 只有当章节有内容 且 (有链接 或 有对应的地点) 时才处理
            # 即使没有 links，如果内容丰富，也可以尝试挖掘（取决于你的策略，这里保守策略是必须有 links 或者是已知地点）
            if section.get("content") and (section.get("links") or loc_ids):
                tasks.append(self._process_single_section(section, loc_ids))

        results = await asyncio.gather(*tasks)

        # 2. 合并结果并去重
        full_graph = {"nodes": [], "edges": []}
        for res in results:
            full_graph["nodes"].extend(res["nodes"])
            full_graph["edges"].extend(res["edges"])

        return self._deduplicate_graph(full_graph)

    async def _process_single_section(self, section: Dict, location_ids: List[str]) -> Dict:
        # 1. 准备候选实体 (从 links 中提取)
        candidates = []

        # 新增一个玩家节点，处理与玩家的互动关系
        candidates.append({
            "tag": "party",
            "text": "The Characters / Party",
            "suggested_id": "party:characters"
        })

        for link in section.get("links", []):
            # 过滤感兴趣的 tag
            if link.get("tag") in ["creature", "item", "spell"]:
                candidates.append({
                    "tag": link["tag"],
                    "text": link["text"],
                    # 预生成一个建议 ID，方便 LLM 参考
                    "suggested_id": f"{link['tag']}:{link['text'].lower().replace(' ', '_')}"
                })
        
        # 如果没有候选实体且没有地点上下文，挖掘价值不大，跳过以节省 Token
        if not candidates and not location_ids:
            return {"nodes": [], "edges": []}

        candidate_list_str = json.dumps(candidates, indent=2, ensure_ascii=False)
        loc_list_str = ", ".join(location_ids)

        # 2. 构建 Prompt (truncate content for smaller models)
        content = section.get("content", "")
        if len(content) > 4000:
            content = content[:4000] + "... [truncated]"

        prompt = PromptFactory.create_entity_enrichment_prompt(
            content,
            candidate_list_str,
            loc_list_str
        )

        # 3. 调用 LLM
        try:
            async with self.semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=4096  # Increased for complex entity extraction with many nodes/edges
                )

                raw_content = response.choices[0].message.content
                try:
                    data = json.loads(raw_content)
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parse error for section {section['id']}: {je}")
                    logger.error(f"Raw response (first 300 chars): {raw_content[:300]}...")
                    return {"nodes": [], "edges": []}

                return data
        except Exception as e:
            logger.error(f"Entity extraction failed for section {section['id']}: {e}")
            return {"nodes": [], "edges": []}

    def _collect_sections(self, nodes: List[Dict]) -> List[Dict]:
        """递归收集所有 Section"""
        collected = []
        for node in nodes:
            collected.append(node)
            if "children" in node:
                collected.extend(self._collect_sections(node["children"]))
        return collected

    def _deduplicate_graph(self, graph: Dict) -> Dict:
        """简单的图谱去重"""
        unique_nodes = {}
        for n in graph["nodes"]:
            if n["id"] not in unique_nodes:
                unique_nodes[n["id"]] = n
            else:
                # 如果新节点信息更全（例如有 label），可以更新
                if len(n.get("label", "")) > len(unique_nodes[n["id"]].get("label", "")):
                    unique_nodes[n["id"]] = n

        unique_edges = []
        seen_edges = set()
        for e in graph["edges"]:
            # 归一化 edge key
            key = f"{e['source']}|{e['relation']}|{e['target']}"
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {"nodes": list(unique_nodes.values()), "edges": unique_edges}