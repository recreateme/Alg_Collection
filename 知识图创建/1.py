from neo4j import GraphDatabase
import time
import traceback


# 1. 加载映射文件
def load_mappings(file_path, encoding="utf8"):
    mapping = {}
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                id, uri = line.strip().split('\t')
                mapping[int(id)] = uri
        return mapping
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return {}


# 2. 加载三元组
def load_triples(file_path):
    triples = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                triples.append((h, r, t))
        print(f"成功加载了 {len(triples)} 个三元组")
        return triples
    except Exception as e:
        print(f"加载三元组文件 {file_path} 时出错: {e}")
        return []


# 3. Neo4j 操作类
class Neo4jLoader:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_nodes(self, entity_map):
        try:
            with self.driver.session() as session:
                session.run("""
                    UNWIND $entities as entity
                    MERGE (n:Entity {uri: entity.uri})
                    SET n.id = entity.id
                """, entities=[{"id": id, "uri": uri} for id, uri in entity_map.items()])
            print("节点创建成功")
        except Exception as e:
            print(f"创建节点失败: {str(e)}")
            raise

    def create_relationships(self, triples, entity_map, relation_map, batch_size=500):
        try:
            with self.driver.session() as session:
                total = len(triples)
                for i in range(0, total, batch_size):
                    batch = triples[i:i + batch_size]
                    params = []
                    for h, r, t in batch:
                        if h not in entity_map or t not in entity_map or r not in relation_map:
                            print(f"跳过无效三元组: ({h}, {r}, {t})")
                            continue
                        params.append({
                            "h_uri": entity_map[h],
                            "t_uri": entity_map[t],
                            "rel_type": relation_map[r].split('/')[-1]
                        })

                    try:
                        session.run("""
                            UNWIND $batch as triple
                            MATCH (h:Entity {uri: triple.h_uri})
                            MATCH (t:Entity {uri: triple.t_uri})
                            MERGE (h)-[r:`{rel_type}`]->(t)
                        """, batch=params)
                        print(f"已处理 {min(i + batch_size, total)}/{total} 个三元组")
                    except Exception as e:
                        print(f"批次 {i} 处理失败: {str(e)}")
                        raise
        except Exception as e:
            print(f"创建关系失败: {str(e)}")
            raise


def main():
    start_time = time.time()
    entity_map = load_mappings('e_map.dat')
    relation_map = load_mappings('r_map.dat')
    triples = load_triples('train.dat')

    if not entity_map or not relation_map or not triples:
        print("数据加载失败，程序退出")
        return

    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "movielens"

    try:
        loader = Neo4jLoader(uri, username, password)

        with loader.driver.session() as session:
            session.run("CREATE INDEX entity_uri IF NOT EXISTS FOR (e:Entity) ON (e.uri)")

        print("开始创建节点...")
        loader.create_nodes(entity_map)
        print("节点创建完成，开始创建关系...")
        loader.create_relationships(triples, entity_map, relation_map, batch_size=500)

        with loader.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            print(f"导入完成！总节点数: {node_count}, 总关系数: {rel_count}")

        loader.close()
        print(f"总耗时: {time.time() - start_time:.2f} 秒")

    except Exception as e:
        print(f"数据库操作出错: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()