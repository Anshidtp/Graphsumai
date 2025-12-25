from neo4j import GraphDatabase
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Construct Neo4j graph from FB15k-237 data"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ Database cleared")
    
    def create_schema(self):
        """Create constraints and indexes"""
        with self.driver.session() as session:
            # Create constraint on Entity.id
            session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            
            # Create index on Entity.name
            session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)
            
            # Create index on Entity.freebase_id
            session.run("""
                CREATE INDEX entity_freebase_id IF NOT EXISTS
                FOR (e:Entity) ON (e.freebase_id)
            """)
        
        logger.info("✅ Schema created")
    
    def batch_create_entities(self, nodes: List[Dict], batch_size: int = 5000):
        """Create entity nodes in batches"""
        total = len(nodes)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = nodes[i:i+batch_size]
                
                query = """
                UNWIND $nodes AS node
                MERGE (e:Entity {id: node.id})
                SET e.name = node.name,
                    e.freebase_id = node.freebase_id,
                    e.type = node.type,
                    e.label = node.label
                """
                
                session.run(query, nodes=batch)
                logger.info(f"Created entities batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
        
        logger.info(f"✅ Created {total} entities")
    
    def batch_create_relationships(self, relationships: List[Dict], batch_size: int = 5000):
        """Create relationships in batches"""
        total = len(relationships)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = relationships[i:i+batch_size]
                
                query = """
                UNWIND $rels AS rel
                MATCH (source:Entity {id: rel.source_id})
                MATCH (target:Entity {id: rel.target_id})
                CALL apoc.create.relationship(source, rel.type, {
                    original_relation: rel.original_relation,
                    readable_name: rel.readable_name
                }, target) YIELD rel as r
                RETURN count(r)
                """
                
                try:
                    session.run(query, rels=batch)
                except Exception as e:
                    # Fallback without APOC
                    logger.warning(f"APOC not available, using MERGE: {e}")
                    query_fallback = """
                    UNWIND $rels AS rel
                    MATCH (source:Entity {id: rel.source_id})
                    MATCH (target:Entity {id: rel.target_id})
                    MERGE (source)-[r:RELATED]->(target)
                    SET r.type = rel.type,
                        r.original_relation = rel.original_relation,
                        r.readable_name = rel.readable_name
                    """
                    session.run(query_fallback, rels=batch)
                
                logger.info(f"Created relationships batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
        
        logger.info(f"✅ Created {total} relationships")
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Node count
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            stats['entity_count'] = result.single()['count']
            
            # Relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['relationship_count'] = result.single()['count']
            
            # Relationship types
            result = session.run("""
                MATCH ()-[r]->()
                RETURN r.readable_name as relation, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            stats['top_relations'] = [dict(record) for record in result]
            
            return stats
    
    def build_graph(self, triples: List[Tuple[str, str, str]], 
                   entities: set, preprocessor, clear: bool = True):
        """Build complete graph from FB15k-237 data"""
        logger.info("🔨 Building FB15k-237 knowledge graph...")
        
        if clear:
            self.clear_database()
        
        # Create schema
        self.create_schema()
        
        # Create nodes
        logger.info("Creating entity nodes...")
        nodes = preprocessor.create_nodes(entities)
        self.batch_create_entities(nodes)
        
        # Create relationships
        logger.info("Creating relationships...")
        relationships = preprocessor.create_relationships(triples)
        self.batch_create_relationships(relationships)
        
        # Get statistics
        stats = self.get_statistics()
        logger.info("✅ Graph construction completed!")
        logger.info(f"Statistics: {stats}")
        
        return stats