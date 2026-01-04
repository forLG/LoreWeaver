"""
Visualize Graph from Neo4j with Custom Queries

Usage:
    python scripts/visualize_from_neo4j.py
"""

import config_neo4j as config
from graph_builder import GraphVisualizer, Neo4jBuilder
from utils.logger import logger


def visualize_creatures_and_locations():
    """可视化生物及其所在位置"""
    builder = Neo4jBuilder(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )

    viz = GraphVisualizer()

    with builder:
        # 查询所有生物及其关系
        query = """
        MATCH (c:Creature)-[r]-(l:Location)
        RETURN c, r, l
        LIMIT 100
        """

        output_path = viz.visualize_from_neo4j(
            builder=builder,
            cypher_query=query,
            output_html="output/visualizations/creature_locations.html",
            title="Creatures and Their Locations"
        )
        logger.info(f"已保存: {output_path}")


def visualize_shortest_path():
    """可视化两点之间的最短路径"""
    builder = Neo4jBuilder(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )

    viz = GraphVisualizer()

    with builder:
        # 查询最短路径
        query = """
        MATCH path = shortestPath(
            (start {id: 'stormwreck_isle'})-[*1..5]-(end {id: 'creature:runara'})
        )
        UNWIND nodes(path) AS n
        UNWIND relationships(path) AS r
        RETURN n, r
        """

        output_path = viz.visualize_from_neo4j(
            builder=builder,
            cypher_query=query,
            output_html="output/visualizations/shortest_path.html",
            title="Shortest Path: Stormwreck Isle → Runara"
        )
        logger.info(f"已保存: {output_path}")


def visualize_filtered_by_type():
    """可视化特定类型的节点"""
    builder = Neo4jBuilder(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )

    viz = GraphVisualizer()

    with builder:
        # 只显示特定类型的位置
        query = """
        MATCH (l:Location)
        WHERE l.type IN ('Cave', 'Cavern', 'Crypts')
        OPTIONAL MATCH (l)-[r:PART_OF*0..2]-(related)
        RETURN l, r, related
        LIMIT 50
        """

        output_path = viz.visualize_from_neo4j(
            builder=builder,
            cypher_query=query,
            output_html="output/visualizations/caves_only.html",
            title="Underground Locations Only"
        )
        logger.info(f"已保存: {output_path}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("从 Neo4j 生成可视化")
    logger.info("=" * 60)

    try:
        logger.info("[1/3] 生物与位置关系...")
        visualize_creatures_and_locations()

        logger.info("[2/3] 最短路径...")
        visualize_shortest_path()

        logger.info("[3/3] 过滤特定类型...")
        visualize_filtered_by_type()

        logger.info("=" * 60)
        logger.info("完成！在浏览器中打开 output/visualizations/ 查看结果")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"错误: {e}")
        logger.info("请确保:")
        logger.info("  1. Neo4j 正在运行")
        logger.info("  2. 已运行 scripts/build_graph.py --mode neo4j 导入数据")
        logger.info("  3. config_neo4j.py 中的密码正确")
