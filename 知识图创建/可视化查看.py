from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import networkx as nx
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnection:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {str(e)}")
            raise

    def close(self):
        self.driver.close()
        logger.info("Neo4j连接已关闭")

    def fetch_graph(self):
        """从Neo4j读取图数据"""
        with self.driver.session() as session:
            query = (
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "RETURN h.name AS head, r.type AS relation, t.name AS tail"
            )
            result = session.run(query)
            return [(record["head"], record["relation"], record["tail"])
                    for record in result]


def build_nx_graph(triples):
    """构建NetworkX图"""
    G = nx.DiGraph()
    for head, relation, tail in triples:
        G.add_node(head)
        G.add_node(tail)
        G.add_edge(head, tail, relation=relation)
    return G


def visualize_graph(G):
    """可视化知识图谱"""
    try:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.show()
        logger.info("图可视化完成")
    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")


def main():
    # 配置参数
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "movielens"

    try:
        conn = Neo4jConnection(uri, user, password)
        triples = conn.fetch_graph()
        graph = build_nx_graph(triples)
        visualize_graph(graph)
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()