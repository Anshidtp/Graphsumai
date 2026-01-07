from neo4j import GraphDatabase
from typing import List, Dict
import logging
from src.k_graph.graph_schema import GraphSchema
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Construct knowledge graph in Neo4j AuraDB"""
    
    def __init__(self, uri: str, username: str, password: str):
        logger.info(f"🔄 Connecting to Neo4j at {uri}")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._verify_connection()
        logger.info("✅ Connected to Neo4j AuraDB")
    
    def _verify_connection(self):
        """Verify database connection"""
        with self.driver.session() as session:
            result = session.run("RETURN 1 as num")
            assert result.single()['num'] == 1
    
    def close(self):
        """Close database connection"""
        self.driver.close()
        logger.info("📤 Disconnected from Neo4j")
    
    def clear_database(self):
        """Clear all data"""
        logger.info("🗑️  Clearing database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ Database cleared")
    
    def create_schema(self):
        """Create schema with constraints and indexes"""
        logger.info("🏗️  Creating schema...")
        
        with self.driver.session() as session:
            # Create constraints
            for query in GraphSchema.get_constraints_queries():
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
            
            # Create vector index
            try:
                session.run(GraphSchema.get_vector_index_query())
                logger.info("✅ Vector index created")
            except Exception as e:
                logger.warning(f"Vector index warning (may already exist): {e}")
        
        logger.info("✅ Schema created")
    
    def batch_create_graph_from_triplets(self, triplets_data: List[Dict], batch_size: int = 1000):
        """
        Create graph from readable triplets
        
        Each triplet creates:
        1. Head entity node (if not exists)
        2. Tail entity node (if not exists)
        3. Relationship between them
        4. Triplet node with embedding
        """
        logger.info(f"📝 Creating graph from {len(triplets_data):,} triplets...")
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(triplets_data), batch_size), desc="Creating graph"):
                batch = triplets_data[i:i+batch_size]
                
                query = """
                UNWIND $triplets AS t
                
                // Create head entity
                MERGE (head:Entity {name: t.head_name})
                ON CREATE SET head.aliases = t.head_aliases
                
                // Create tail entity
                MERGE (tail:Entity {name: t.tail_name})
                ON CREATE SET tail.aliases = t.tail_aliases
                
                // Create relationship (use relation as type, but sanitized)
                CALL apoc.create.relationship(head, t.relation_type, {
                    relation: t.relation,
                    readable: t.relation
                }, tail) YIELD rel
                
                // Create triplet node with embedding
                CREATE (trip:Triplet {
                    text: t.triplet_text,
                    embedding: t.embedding
                })
                
                // Connect triplet to entities
                CREATE (trip)-[:HAS_HEAD]->(head)
                CREATE (trip)-[:HAS_TAIL]->(tail)
                
                RETURN count(*) as created
                """
                
                try:
                    session.run(query, triplets=batch)
                except Exception as e:
                    # Fallback: create simple relationships without apoc
                    logger.warning(f"APOC not available, using simple relationships: {e}")
                    
                    simple_query = """
                    UNWIND $triplets AS t
                    
                    MERGE (head:Entity {name: t.head_name})
                    ON CREATE SET head.aliases = t.head_aliases
                    
                    MERGE (tail:Entity {name: t.tail_name})
                    ON CREATE SET tail.aliases = t.tail_aliases
                    
                    MERGE (head)-[r:RELATED_TO]->(tail)
                    SET r.relation = t.relation,
                        r.readable = t.relation
                    
                    CREATE (trip:Triplet {
                        text: t.triplet_text,
                        embedding: t.embedding
                    })
                    
                    CREATE (trip)-[:HAS_HEAD]->(head)
                    CREATE (trip)-[:HAS_TAIL]->(tail)
                    """
                    
                    session.run(simple_query, triplets=batch)
        
        logger.info("✅ Graph created successfully")
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Entity count
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats['entities'] = result.single()['count']
            
            # Triplet count
            result = session.run("MATCH (t:Triplet) RETURN count(t) as count")
            stats['triplets'] = result.single()['count']
            
            # Relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['relationships'] = result.single()['count']
            
            # Sample entities
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.name as name
                ORDER BY rand()
                LIMIT 5
            """)
            stats['sample_entities'] = [record['name'] for record in result]
            
            return stats
