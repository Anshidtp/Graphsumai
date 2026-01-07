from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class QueryEngine:
    """Query engine for knowledge graph"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search entities by name or aliases"""
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($term)
           OR any(alias IN e.aliases WHERE alias CONTAINS toLower($term))
        RETURN e.name as name, e.aliases as aliases
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, term=search_term, limit=limit)
            return [dict(record) for record in result]
    
    def get_entity_triplets(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """Get all triplets involving an entity"""
        query = """
        MATCH (e:Entity {name: $entity_name})
        MATCH (t:Triplet)-[:HAS_HEAD|HAS_TAIL]->(e)
        RETURN t.text as triplet_text
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name, limit=limit)
            return [dict(record) for record in result]
    
    def vector_search_triplets(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Vector similarity search on triplet embeddings"""
        query = """
        CALL db.index.vector.queryNodes('triplet_embeddings', $limit, $embedding)
        YIELD node, score
        RETURN node.text as triplet_text, score
        ORDER BY score DESC
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(query, embedding=query_embedding, limit=limit)
                return [dict(record) for record in result]
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                return []
    
    def get_related_entities(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """Get entities related to given entity"""
        query = """
        MATCH (e:Entity {name: $entity_name})-[r]-(related:Entity)
        RETURN DISTINCT related.name as name, type(r) as relation, r.readable as readable
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name, limit=limit)
            return [dict(record) for record in result]