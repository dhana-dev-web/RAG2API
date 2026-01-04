from neo4j import GraphDatabase

class Neo4jStore:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def add_relation(self, e1, relation, e2):
        query = """
        MERGE (a:Entity {name: $e1})
        MERGE (b:Entity {name: $e2})
        MERGE (a)-[r:RELATED_TO]->(b)
        """
        with self.driver.session() as session:
            session.run(query, e1=e1, e2=e2)

    def get_entities(self):
        query = "MATCH (n:Entity) RETURN n.name AS name"
        with self.driver.session() as session:
            return [r["name"] for r in session.run(query)]

    def get_relations(self):
        query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        """
        with self.driver.session() as session:
            return list(session.run(query))

    def query_entity(self, entity):
        query = """
        MATCH (a:Entity {name:$name})-[r]->(b)
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        """
        with self.driver.session() as session:
            return list(session.run(query, name=entity))
