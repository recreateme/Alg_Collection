from neo4j import GraphDatabase
from tqdm import tqdm
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

    def create_batch(self, triples, batch_size=1000):
        """批量创建节点和关系"""
        with self.driver.session() as session:
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                try:
                    tx = session.begin_transaction()
                    for head, relation, tail in batch:
                        query = (
                            "MERGE (h:Entity {name: $head}) "
                            "MERGE (t:Entity {name: $tail}) "
                            "MERGE (h)-[r:RELATION {type: $relation}]->(t)"
                        )
                        tx.run(query, head=head, relation=relation, tail=tail)
                    tx.commit()
                    logger.info(f"成功提交 {len(batch)} 个三元组")
                except Exception as e:
                    tx.rollback()
                    logger.error(f"批量处理失败: {str(e)}")


def build_kg_from_file(file_path, neo4j_conn, max_triples=None):
    """从文件构建知识图谱"""
    triples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            total_lines = len(lines) if not max_triples else min(len(lines), max_triples)

            with tqdm(total=total_lines, desc="处理三元组") as pbar:
                for i, line in enumerate(lines):
                    if max_triples and i >= max_triples:
                        break
                    try:
                        head, tail, relation = line.strip().split('\t')
                        triples.append((head, relation, tail))
                    except ValueError as e:
                        logger.warning(f"跳过无效行 {i + 1}: {line.strip()}")
                    pbar.update(1)

        neo4j_conn.create_batch(triples)
        logger.info(f"总共处理 {len(triples)} 个三元组")

    except Exception as e:
        logger.error(f"构建知识图谱失败: {str(e)}")
        raise


def main():
    # 配置参数
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "movielens"
    file_path = "train.txt"

    try:
        conn = Neo4jConnection(uri, user, password)
        build_kg_from_file(file_path, conn)
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()