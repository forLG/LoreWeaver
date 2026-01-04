"""
LoreWeaver Graph Building Script

功能：
1. 将 JSON 图谱导入 Neo4j 数据库
2. 生成交互式 HTML 可视化
3. 查询图谱统计信息

使用方法：
    python scripts/build_graph.py --mode neo4j          # 导入到 Neo4j
    python scripts/build_graph.py --mode visualize      # 仅生成可视化
    python scripts/build_graph.py --mode both           # 同时执行
    python scripts/build_graph.py --mode clear          # 清空数据库

环境要求：
    - Neo4j 数据库运行中
    - 已安装 neo4j-python-driver
"""
import argparse
import sys

import config_neo4j as config
from graph_builder import GraphVisualizer, Neo4jBuilder
from utils.logger import logger


def build_neo4j_graph():
    """构建 Neo4j 图谱"""
    logger.info("=" * 60)
    logger.info("开始构建 Neo4j 图谱...")
    logger.info("=" * 60)

    # 检查输入文件
    if not config.LOCATION_GRAPH_FILE.exists():
        logger.error(f"找不到位置图谱文件 {config.LOCATION_GRAPH_FILE}")
        logger.info("请先运行 python -m main --stage all 生成图谱数据")
        return False

    if not config.ENTITY_GRAPH_FILE.exists():
        logger.error(f"找不到实体图谱文件 {config.ENTITY_GRAPH_FILE}")
        logger.info("请先运行 python -m main --stage all 生成图谱数据")
        return False

    # 创建建图器
    builder = Neo4jBuilder(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
        batch_size=config.BATCH_SIZE
    )

    try:
        with builder:
            # 创建约束
            logger.info("[步骤 1/4] 创建数据库约束...")
            builder.create_constraints()

            # 导入位置图谱
            logger.info(f"[步骤 2/4] 导入位置图谱: {config.LOCATION_GRAPH_FILE}")
            loc_stats = builder.import_location_graph(str(config.LOCATION_GRAPH_FILE))
            logger.info(f"  -> 节点: {loc_stats['nodes']}, 关系: {loc_stats['edges']}")

            # 导入实体图谱
            logger.info(f"[步骤 3/4] 导入实体图谱: {config.ENTITY_GRAPH_FILE}")
            ent_stats = builder.import_entity_graph(str(config.ENTITY_GRAPH_FILE))
            logger.info(f"  -> 节点: {ent_stats['nodes']}, 关系: {ent_stats['edges']}")

            # 统计信息
            logger.info("[步骤 4/4] 图谱统计:")
            stats = builder.get_stats()
            logger.info(f"  总节点数: {stats['total_nodes']}")
            logger.info(f"  总关系数: {stats['total_relationships']}")
            logger.info("  节点类型分布:")
            for label, count in stats['nodes'].items():
                logger.info(f"    {label}: {count}")
            logger.info("  关系类型分布:")
            for rel_type, count in stats['relationships'].items():
                logger.info(f"    {rel_type}: {count}")

            # 示例查询
            logger.info("[示例查询] 查找最短路径示例:")
            path_examples = [
                ("stormwreck_isle", "creature:runara"),
                ("the_wreck_of_the_compass_rose", "clifftop_observatory_tower"),
            ]
            for start, end in path_examples:
                path = builder.find_shortest_path(start, end)
                if path:
                    logger.info(f"  {start} -> {end}: {len(path['nodes'])} 跳")

        logger.info("=" * 60)
        logger.info("Neo4j 图谱构建完成!")
        logger.info("=" * 60)
        logger.info("访问 Neo4j Browser: http://localhost:7474")
        logger.info("示例查询:")
        logger.info("  MATCH (n:Creature) RETURN n LIMIT 25")
        logger.info("  MATCH (l:Location {id: 'stormwreck_isle'})-[r:PART_OF*0..3]-(sub) RETURN sub")
        return True

    except Exception as e:
        logger.error(f"错误: {e}")
        logger.info("请检查:")
        logger.info("  1. Neo4j 是否正在运行")
        logger.info("  2. 配置文件 config_neo4j.py 中的连接信息是否正确")
        logger.info(f"  3. 密码是否正确 (当前: {config.NEO4J_PASSWORD})")
        return False


def generate_visualizations():
    """生成可视化 HTML 文件"""
    logger.info("=" * 60)
    logger.info("生成图谱可视化...")
    logger.info("=" * 60)

    # 检查输入文件
    if not config.LOCATION_GRAPH_FILE.exists():
        logger.error("找不到位置图谱文件")
        return False

    if not config.ENTITY_GRAPH_FILE.exists():
        logger.error("找不到实体图谱文件")
        return False

    # 创建可视化器
    viz = GraphVisualizer(**config.VISUALIZATION_CONFIG)

    # 1. 位置图谱可视化
    logger.info("[1/3] 生成位置图谱可视化...")
    loc_path = viz.visualize_from_json(
        str(config.LOCATION_GRAPH_FILE),
        output_html=str(config.VISUALIZATION_DIR / "location_graph.html"),
        title="Location Graph - Stormwreck Isle"
    )
    logger.info(f"  -> {loc_path}")

    # 2. 实体图谱可视化（只显示生物）
    logger.info("[2/3] 生成实体图谱可视化（生物 + 位置）...")
    ent_path = viz.visualize_from_json(
        str(config.ENTITY_GRAPH_FILE),
        output_html=str(config.VISUALIZATION_DIR / "entity_graph.html"),
        node_filter=['Creature', 'Location'],
        max_nodes=150,
        title="Entity Graph - Creatures & Locations"
    )
    logger.info(f"  -> {ent_path}")

    # 3. 联合可视化
    logger.info("[3/3] 生成联合可视化...")
    combined_path = viz.visualize_combined(
        str(config.LOCATION_GRAPH_FILE),
        str(config.ENTITY_GRAPH_FILE),
        output_html=str(config.VISUALIZATION_DIR / "combined_graph.html"),
        max_nodes=200
    )
    logger.info(f"  -> {combined_path}")

    logger.info("=" * 60)
    logger.info("可视化生成完成!")
    logger.info("=" * 60)
    logger.info("在浏览器中打开以下文件查看:")
    for f in [loc_path, ent_path, combined_path]:
        path_str = f.replace('\\', '/')
        logger.info(f"  file:///{path_str}")
    return True


def clear_neo4j_graph():
    """清空 Neo4j 图谱"""
    logger.info("=" * 60)
    logger.info("清空 Neo4j 图谱")
    logger.info("=" * 60)

    confirm = input("确认要清空所有图谱数据吗? (输入 'yes' 确认): ")
    if confirm.lower() != 'yes':
        logger.info("操作已取消")
        return

    builder = Neo4jBuilder(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )

    with builder:
        builder.clear_graph(confirm=True)

    logger.info("图谱已清空")


def main():
    parser = argparse.ArgumentParser(description="LoreWeaver 图谱构建工具")
    parser.add_argument(
        '--mode',
        choices=['neo4j', 'visualize', 'both', 'clear'],
        default='visualize',
        help='操作模式: neo4j(建图), visualize(可视化), both(两者), clear(清空)'
    )

    args = parser.parse_args()

    if args.mode == 'neo4j':
        success = build_neo4j_graph()
        sys.exit(0 if success else 1)

    elif args.mode == 'visualize':
        success = generate_visualizations()
        sys.exit(0 if success else 1)

    elif args.mode == 'both':
        success1 = build_neo4j_graph()
        success2 = generate_visualizations()
        sys.exit(0 if (success1 and success2) else 1)

    elif args.mode == 'clear':
        clear_neo4j_graph()


if __name__ == "__main__":
    main()
