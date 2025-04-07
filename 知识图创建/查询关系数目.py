from neo4j import GraphDatabase


# Neo4j 连接类
class Neo4jConnection:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def get_relation_types_count(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS relation_type, count(r) AS count
                ORDER BY count DESC
            """)
            relation_counts = [(record["relation_type"], record["count"]) for record in result]
            return relation_counts


def main():
    # 连接参数
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "movielens"

    # 查询关系类型
    conn = Neo4jConnection(uri, username, password)
    relation_counts = conn.get_relation_types_count()
    conn.close()

    # 输出结果
    print(f"总关系类型数: {len(relation_counts)}")
    print("各关系类型及数量:")
    for rel_type, count in relation_counts:
        print(f"{rel_type}: {count}")


if __name__ == "__main__":
    main()