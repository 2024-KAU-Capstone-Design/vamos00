from neo4j import GraphDatabase

import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j 설정
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_database = os.getenv("NEO4J_DATABASE")

# 드라이버 생성
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    

def clear_database():
    with driver.session(database=neo4j_database) as session:
        # Step 1: 모든 노드의 속성을 삭제
        session.run("MATCH (n) SET n = {}")

        # Step 2: 모든 노드 및 관계 삭제
        session.run("MATCH (n) DETACH DELETE n")


try:
    clear_database()
    print("Neo4j 데이터베이스가 초기화되었습니다.")
finally:
    driver.close()
