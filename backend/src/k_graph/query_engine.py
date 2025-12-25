from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class QueryEngine:
    """Query engine for FB15k-237 knowledge graph"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def find_entity_by_name(self, name: str, limit: int = 10) -> List[Dict]:
        """Find entities by name"""
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $name OR e.freebase_id CONTAINS $name
            RETURN e
            LIMIT $limit
            """
            result = session.run(query, name=name, limit=limit)
            return [dict(record['e']) for record in result]
    
    def get_entity_neighbors(self, entity_id: str, depth: int = 1) -> List[Dict]:
        """Get neighbors of an entity"""
        with self.driver.session() as session:
            query = f"""
            MATCH path = (e:Entity {{id: $entity_id}})-[*1..{depth}]-(neighbor)
            RETURN DISTINCT neighbor, length(path) as distance
            ORDER BY distance
            LIMIT 50
            """
            result = session.run(query, entity_id=entity_id)
            return [dict(record) for record in result]
    
    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[Dict]:
        """Find path between two entities"""
        with self.driver.session() as session:
            query = f"""
            MATCH (start:Entity {{id: $start_id}}), (end:Entity {{id: $end_id}})
            MATCH path = shortestPath((start)-[*..{max_depth}]-(end))
            RETURN nodes(path) as nodes, relationships(path) as rels
            """
            result = session.run(query, start_id=start_id, end_id=end_id)
            return [dict(record) for record in result]
    
    def get_subgraph_for_query(self, query_text: str, k: int = 10) -> str:
        """Get subgraph relevant to query"""
        # Search for relevant entities
        with self.driver.session() as session:
            search_query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query OR e.freebase_id CONTAINS $query
            WITH e
            LIMIT $k
            OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
            RETURN e, r, neighbor
            LIMIT 100
            """
            
            result = session.run(search_query, query=query_text, k=k)
            
            # Format results
            context_parts = []
            for record in result:
                entity = record['e']
                context_parts.append(f"Entity: {entity['name']} ({entity['id']})")
                
                if record['r'] and record['neighbor']:
                    rel = record['r']
                    neighbor = record['neighbor']
                    context_parts.append(
                        f"  - {rel.get('readable_name', 'RELATED')} -> {neighbor['name']}"
                    )
            
            return "\n".join(context_parts[:100])