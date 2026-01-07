from typing import List


class GraphSchema:
    """Define schema for FB15k-237 knowledge graph"""
    
    @staticmethod
    def get_constraints_queries() -> List[str]:
        """Get Cypher queries to create constraints"""
        return [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_aliases IF NOT EXISTS FOR (e:Entity) ON (e.aliases)",
        ]
    
    @staticmethod
    def get_vector_index_query() -> str:
        """Get Cypher query to create vector index for triplet embeddings"""
        return """
        CALL db.index.vector.createNodeIndex(
            'triplet_embeddings',
            'Triplet',
            'embedding',
            384,
            'cosine'
        )
        """