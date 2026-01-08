# from typing import List, Dict
# import logging

# logger = logging.getLogger(__name__)


# class KnowledgeGraphRetriever:
#     """Retrieve relevant context from knowledge graph"""
    
#     def __init__(self, query_engine, embedder, max_neighbors: int = 50):
#         self.query_engine = query_engine
#         self.embedder = embedder
#         self.max_neighbors = max_neighbors
    
#     def retrieve(self, query: str, top_k: int = 10, depth: int = 2) -> Dict:
#         """
#         Retrieve relevant context for a query
#         """
#         logger.info(f"🔍 Retrieving context for: {query}")
        
#         # Generate query embedding
#         query_embedding = self.embedder.embed_text(query)[0].tolist()
        
#         # Hybrid search
#         search_results = self.query_engine.hybrid_search(
#             query, query_embedding, limit=top_k
#         )
        
#         if not search_results:
#             logger.warning("No entities found")
#             return {
#                 'entities': [],
#                 'context': "No relevant information found in the knowledge graph.",
#                 'metadata': {'query': query, 'found': 0}
#             }
        
#         # Get top entities
#         top_entities = [r['entity'] for r in search_results]
#         entity_ids = [e['id'] for e in top_entities]
        
#         # Expand to neighbors
#         context_data = self._expand_context(entity_ids, depth)
        
#         # Format context
#         formatted_context = self._format_context(context_data)
        
#         logger.info(f"✅ Retrieved context with {len(context_data['entities'])} entities")
        
#         return {
#             'entities': context_data['entities'],
#             'relationships': context_data['relationships'],
#             'context': formatted_context,
#             'metadata': {
#                 'query': query,
#                 'found': len(top_entities),
#                 'expanded_to': len(context_data['entities'])
#             }
#         }
    
#     def _expand_context(self, entity_ids: List[str], depth: int) -> Dict:
#         """Expand context by traversing graph"""
#         all_entities = {}
#         all_relationships = []
        
#         for entity_id in entity_ids:
#             # Get neighbors
#             neighbors = self.query_engine.get_multi_hop_neighbors(
#                 entity_id, hops=depth, limit=self.max_neighbors
#             )
            
#             for neighbor_data in neighbors:
#                 neighbor = neighbor_data['neighbor']
#                 neighbor_id = neighbor['id']
                
#                 if neighbor_id not in all_entities:
#                     all_entities[neighbor_id] = neighbor
                
#                 all_relationships.append({
#                     'source': entity_id,
#                     'target': neighbor_id,
#                     'distance': neighbor_data['distance']
#                 })
        
#         return {
#             'entities': list(all_entities.values()),
#             'relationships': all_relationships
#         }
    
#     def _format_context(self, context_data: Dict) -> str:
#         """Format context as readable text"""
#         entities = context_data['entities']
        
#         if not entities:
#             return "No relevant information found."
        
#         # Build context string
#         lines = ["# Knowledge Graph Context\n"]
        
#         # Add entity information
#         lines.append("## Relevant Entities:")
#         for entity in entities[:15]:  # Top 15 entities
#             name = entity.get('name', 'Unknown')
#             description = entity.get('description', '')
            
#             lines.append(f"\n**{name}**")
#             if description:
#                 lines.append(f"  {description}")
#             lines.append(f"  Connections: {entity.get('degree', 0)}")
        
#         return "\n".join(lines)

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphRetriever:
    """Retrieve relevant context from knowledge graph using triplet embeddings"""
    
    def __init__(self, query_engine, embedder):
        self.query_engine = query_engine
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 10) -> Dict:
        """
        Retrieve relevant triplets for a query
        
        Strategy:
        1. Generate query embedding
        2. Vector search for similar triplets
        3. Extract entities from top triplets
        4. Expand with related triplets
        """
        logger.info(f"🔍 Retrieving context for: {query}")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_triplet(query).tolist()
        
        # Vector search for similar triplets
        triplet_results = self.query_engine.vector_search_triplets(
            query_embedding, 
            limit=top_k * 2
        )
        
        if not triplet_results:
            logger.warning("No triplets found")
            return {
                'triplets': [],
                'context': "No relevant information found in the knowledge graph.",
                'metadata': {'query': query, 'found': 0}
            }
        
        # Extract unique triplets
        triplets = []
        seen = set()
        
        for result in triplet_results:
            text = result['triplet_text']
            if text not in seen:
                triplets.append({
                    'text': text,
                    'score': result.get('score', 0)
                })
                seen.add(text)
        
        triplets = triplets[:top_k]
        
        # Format context
        formatted_context = self._format_context(triplets)
        
        logger.info(f"✅ Retrieved {len(triplets)} relevant triplets")
        
        return {
            'triplets': triplets,
            'context': formatted_context,
            'metadata': {
                'query': query,
                'found': len(triplets)
            }
        }
    
    def _format_context(self, triplets: List[Dict]) -> str:
        """Format triplets as readable context"""
        if not triplets:
            return "No relevant information found."
        
        lines = ["# Knowledge Graph Context\n"]
        lines.append("## Relevant Facts:\n")
        
        for i, triplet in enumerate(triplets, 1):
            text = triplet['text']
            score = triplet.get('score', 0)
            lines.append(f"{i}. {text} (relevance: {score:.3f})")
        
        return "\n".join(lines)