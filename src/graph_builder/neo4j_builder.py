"""
Neo4j Graph Builder for LoreWeaver

使用 neo4j-python-driver 直接将 JSON 图谱导入 Neo4j 数据库。
"""
import json
import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Result

logger = logging.getLogger(__name__)


class Neo4jBuilder:
    """
    Neo4j 建图器 - 使用 Bolt 协议直接写入数据库

    特性：
    - 以 id 为节点的唯一标识
    - 使用 MERGE 确保幂等性（可重复运行）
    - 批量事务处理提高性能
    - 自动创建约束和索引
    """

    # 实体类型到 Neo4j 标签的映射
    TYPE_LABEL_MAP = {
        'creature': 'Creature',
        'item': 'Item',
        'spell': 'Spell',
        'location': 'Location',
        'party': 'Party',
        'unknown': 'Entity',
    }

    # 关系类型映射（统一为大写下划线格式）
    RELATION_MAP = {
        'part_of': 'PART_OF',
        'contains': 'CONTAINS',
        'inhabits': 'INHABITS',
        'explores': 'EXPLORES',
        'knows': 'KNOWS',
        'guards': 'GUARDS',
        'found_in': 'FOUND_IN',
        'drops': 'DROPS',
        'uses': 'USES',
        'located_at': 'LOCATED_AT',
    }

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        batch_size: int = 500,
        verbose: bool = True
    ):
        """
        初始化 Neo4j 建图器

        Args:
            uri: Neo4j 连接 URI (如 bolt://localhost:7687)
            user: 数据库用户名
            password: 数据库密码
            database: 数据库名称（Neo4j 4.0+ 支持多数据库）
            batch_size: 批量处理的节点/边数量
            verbose: 是否打印详细日志
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        self.verbose = verbose

        self._driver = None
        self._connected = False

    # ---------------------------------------------------------------------
    # 连接管理
    # ---------------------------------------------------------------------

    def connect(self) -> None:
        """建立数据库连接"""
        if self._connected:
            return

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 验证连接
            self._driver.verify_connectivity()
            self._connected = True
            if self.verbose:
                logger.info(f"已连接到 Neo4j: {self.uri}")
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise

    def close(self) -> None:
        """关闭数据库连接"""
        if self._driver:
            self._driver.close()
            self._connected = False
            if self.verbose:
                logger.info("数据库连接已关闭")

    def __enter__(self):
        """支持 with 语句"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.close()

    # ---------------------------------------------------------------------
    # 数据库初始化
    # ---------------------------------------------------------------------

    def create_constraints(self) -> None:
        """
        创建唯一性约束和索引

        约束说明：
        - Location.id: 唯一标识位置节点
        - Creature.id: 唯一标识生物节点
        - Item.id: 唯一标识物品节点
        - Spell.id: 唯一标识法术节点

        注意：必须在导入数据前执行，否则 MERGE 性能会很差
        """
        constraints = [
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT creature_id IF NOT EXISTS FOR (c:Creature) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT spell_id IF NOT EXISTS FOR (s:Spell) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT party_id IF NOT EXISTS FOR (p:Party) REQUIRE p.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX entity_label_idx IF NOT EXISTS FOR (n:Creature|Item|Spell) ON (n.label)",
        ]

        with self._driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    if self.verbose:
                        logger.info(f"创建约束: {constraint.split()[-2]}")
                except Exception as e:
                    logger.warning(f"约束可能已存在: {e}")

            for idx in indexes:
                try:
                    session.run(idx)
                    if self.verbose:
                        logger.info(f"创建索引: {idx.split()[-2]}")
                except Exception as e:
                    logger.warning(f"索引可能已存在: {e}")

    def clear_graph(self, confirm: bool = False) -> None:
        """
        清空图谱（开发环境使用）

        Args:
            confirm: 必须为 True 才执行删除操作
        """
        if not confirm:
            logger.warning("清空操作未执行，请设置 confirm=True")
            return

        with self._driver.session(database=self.database) as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count = result.single()["count"]
            if self.verbose:
                logger.info(f"正在删除 {count} 个节点...")
            session.run("MATCH (n) DETACH DELETE n")
            if self.verbose:
                logger.info("图谱已清空")

    # ---------------------------------------------------------------------
    # 位置图谱导入
    # ---------------------------------------------------------------------

    def import_location_graph(
        self,
        location_graph_file: str,
        clear_existing: bool = False
    ) -> Dict[str, int]:
        """
        导入位置图谱

        Args:
            location_graph_file: location_graph.json 文件路径
            clear_existing: 是否先清空现有的 Location 节点

        Returns:
            统计信息字典 {'nodes': int, 'edges': int}
        """
        if self.verbose:
            logger.info(f"导入位置图谱: {location_graph_file}")

        with open(location_graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        if self.verbose:
            logger.info(f"读取到 {len(nodes)} 个节点, {len(edges)} 条边")

        if clear_existing:
            self._clear_nodes_by_label(['Location'])

        # 创建节点
        self._create_location_nodes(nodes)

        # 创建关系
        self._create_location_edges(edges)

        return {
            'nodes': len(nodes),
            'edges': len(edges)
        }

    def _create_location_nodes(self, nodes: List[Dict]) -> None:
        """批量创建位置节点（使用 MERGE 保证幂等性）"""

        def create_batch(tx, batch):
            query = """
            UNWIND $nodes AS node
            MERGE (l:Location {id: node.id})
            SET l.label = node.label,
                l.type = node.type
            """
            tx.run(query, nodes=batch)

        with self._driver.session(database=self.database) as session:
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i + self.batch_size]
                session.execute_write(create_batch, batch)
                if self.verbose:
                    logger.info(f"位置节点: {min(i + self.batch_size, len(nodes))}/{len(nodes)}")

    def _create_location_edges(self, edges: List[Dict]) -> None:
        """批量创建位置关系"""

        def create_batch(tx, batch):
            # 动态构建关系类型
            query = """
            UNWIND $edges AS edge
            MATCH (s:Location {id: edge.source})
            MATCH (t:Location {id: edge.target})
            CALL apoc.create.relationship(s, edge.relation_type, {}, t)
            YIELD rel
            RETURN count(rel)
            """
            # 如果没有 APOC，使用以下替代方案：
            query_alt = """
            UNWIND $edges AS edge
            MATCH (s:Location {id: edge.source})
            MATCH (t:Location {id: edge.target})
            CALL apoc.do.when(
                edge.relation_type = 'PART_OF',
                'MERGE (s)-[r:PART_OF]->(t) RETURN r',
                'MERGE (s)-[r:CONNECTED_TO]->(t) RETURN r',
                {s: s, t: t}
            ) YIELD r
            RETURN count(r)
            """
            # 尝试使用 APOC，如果失败则手动构建
            try:
                tx.run(query, edges=batch)
            except Exception:
                # 不使用 APOC，手动构建每个关系
                for edge in batch:
                    rel_type = self._normalize_relation_type(edge.get('relation', 'CONNECTED_TO'))
                    tx.run(f"""
                        MATCH (s:Location {{id: $source}})
                        MATCH (t:Location {{id: $target}})
                        MERGE (s)-[r:{rel_type}]->(t)
                    """, source=edge['source'], target=edge['target'])

        with self._driver.session(database=self.database) as session:
            for i in range(0, len(edges), self.batch_size):
                batch = edges[i:i + self.batch_size]
                session.execute_write(create_batch, batch)
                if self.verbose:
                    logger.info(f"位置关系: {min(i + self.batch_size, len(edges))}/{len(edges)}")

    # ---------------------------------------------------------------------
    # 实体图谱导入
    # ---------------------------------------------------------------------

    def import_entity_graph(
        self,
        entity_graph_file: str,
        clear_existing: bool = False
    ) -> Dict[str, int]:
        """
        导入实体图谱

        Args:
            entity_graph_file: entity_graph.json 文件路径
            clear_existing: 是否先清空现有的实体节点

        Returns:
            统计信息字典
        """
        if self.verbose:
            logger.info(f"导入实体图谱: {entity_graph_file}")

        with open(entity_graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        if self.verbose:
            logger.info(f"读取到 {len(nodes)} 个节点, {len(edges)} 条边")

        if clear_existing:
            self._clear_nodes_by_label(['Creature', 'Item', 'Spell', 'Party'])

        # 按类型分组节点
        nodes_by_type = self._group_nodes_by_type(nodes)

        # 创建各类型节点
        for node_type, type_nodes in nodes_by_type.items():
            self._create_entity_nodes(type_nodes, node_type)

        # 创建关系
        self._create_entity_edges(edges)

        return {
            'nodes': len(nodes),
            'edges': len(edges)
        }

    def _group_nodes_by_type(self, nodes: List[Dict]) -> Dict[str, List[Dict]]:
        """按类型分组节点"""
        grouped = {}
        for node in nodes:
            node_type = node.get('type', 'unknown').lower()
            if node_type not in grouped:
                grouped[node_type] = []
            grouped[node_type].append(node)
        if self.verbose:
            logger.info(f"节点类型分布: {[(k, len(v)) for k, v in grouped.items()]}")
        return grouped

    def _create_entity_nodes(self, nodes: List[Dict], node_type: str) -> None:
        """批量创建指定类型的实体节点"""

        label = self.TYPE_LABEL_MAP.get(node_type.lower(), 'Entity')

        def create_batch(tx, batch):
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{label} {{id: node.id}})
            SET n.label = node.label
            """
            tx.run(query, nodes=batch)

        with self._driver.session(database=self.database) as session:
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i + self.batch_size]
                session.execute_write(create_batch, batch)
                if self.verbose:
                    logger.info(f"{label} 节点: {min(i + self.batch_size, len(nodes))}/{len(nodes)}")

    def _create_entity_edges(self, edges: List[Dict]) -> None:
        """批量创建实体关系"""

        def create_batch(tx, batch):
            for edge in batch:
                rel_type = self._normalize_relation_type(edge.get('relation', 'RELATED_TO'))
                query = f"""
                MATCH (s), (t) WHERE s.id = $source AND t.id = $target
                MERGE (s)-[r:{rel_type}]->(t)
                """
                params = {
                    'source': edge['source'],
                    'target': edge['target']
                }
                # 添加可选的描述属性
                if 'desc' in edge:
                    query += " SET r.desc = $desc"
                    params['desc'] = edge['desc']

                tx.run(query, **params)

        with self._driver.session(database=self.database) as session:
            for i in range(0, len(edges), self.batch_size):
                batch = edges[i:i + self.batch_size]
                session.execute_write(create_batch, batch)
                if self.verbose:
                    logger.info(f"实体关系: {min(i + self.batch_size, len(edges))}/{len(edges)}")

    # ---------------------------------------------------------------------
    # 查询与验证
    # ---------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        with self._driver.session(database=self.database) as session:
            # 节点统计
            node_result = session.run("""
                MATCH (n)
                RETURN labels(n) AS labels, count(n) AS count
                ORDER BY count DESC
            """)
            node_stats = {row['labels'][0]: row['count'] for row in node_result}

            # 关系统计
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """)
            rel_stats = {row['type']: row['count'] for row in rel_result}

            return {
                'nodes': node_stats,
                'relationships': rel_stats,
                'total_nodes': sum(node_stats.values()),
                'total_relationships': sum(rel_stats.values())
            }

    def find_shortest_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> List[Dict]:
        """查找两个节点之间的最短路径"""
        with self._driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (start {id: $start})-[*1..{max_depth}]-(end {id: $end})
                )
                RETURN [node in nodes(path) | {{
                    id: node.id,
                    label: node.label,
                    labels: labels(node)
                }}] AS nodes,
                [rel in relationships(path) | {{
                    type: type(rel),
                    source: startNode(rel).id,
                    target: endNode(rel).id
                }}] AS relationships
            """, start=start_id, end=end_id, max_depth=max_depth)

            record = result.single()
            if record:
                return {
                    'nodes': record['nodes'],
                    'relationships': record['relationships']
                }
            return None

    # ---------------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------------

    def _normalize_relation_type(self, relation: str) -> str:
        """规范化关系类型名称（大写、下划线）"""
        normalized = relation.upper().replace(' ', '_').replace('-', '_')
        return self.RELATION_MAP.get(relation.lower(), normalized)

    def _clear_nodes_by_label(self, labels: List[str]) -> None:
        """删除指定标签的所有节点"""
        with self._driver.session(database=self.database) as session:
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                count = result.single()["count"]
                if count > 0:
                    if self.verbose:
                        logger.info(f"删除 {count} 个 {label} 节点")
                    session.run(f"MATCH (n:{label}) DETACH DELETE n")


# ---------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # 基本使用
    builder = Neo4jBuilder(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password"
    )

    # 使用 with 语句自动管理连接
    with builder:
        # 1. 创建约束
        builder.create_constraints()

        # 2. 导入位置图谱
        location_stats = builder.import_location_graph(
            "output/location_graph.json"
        )
        print(f"位置图谱: {location_stats}")

        # 3. 导入实体图谱
        entity_stats = builder.import_entity_graph(
            "output/entity_graph.json"
        )
        print(f"实体图谱: {entity_stats}")

        # 4. 查看统计
        stats = builder.get_stats()
        print(f"\n图谱统计:")
        print(f"  节点: {stats['total_nodes']}")
        print(f"  关系: {stats['total_relationships']}")
        print(f"  节点类型: {stats['nodes']}")
        print(f"  关系类型: {stats['relationships']}")

        # 5. 查询示例：最短路径
        path = builder.find_shortest_path("stormwreck_isle", "creature:runara")
        if path:
            print(f"\n最短路径包含 {len(path['nodes'])} 个节点")
