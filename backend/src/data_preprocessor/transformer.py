from typing import List, Tuple, Dict
import numpy as np
from .entity_resolver import EntityResolver

class DataPreprocessor:
    """Preprocess FB15k-237 data for graph construction"""
    
    def __init__(self, resolver: EntityResolver):
        self.resolver = resolver
    
    def create_nodes(self, entities: set) -> List[Dict]:
        """Create node records from entities"""
        nodes = []
        
        for entity_id in entities:
            node = {
                'id': entity_id,
                'label': 'Entity',
                'name': self.resolver.resolve_entity(entity_id),
                'freebase_id': entity_id,
                'type': self._infer_entity_type(entity_id)
            }
            nodes.append(node)
        
        return nodes
    
    def create_relationships(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
        """Create relationship records from triples"""
        relationships = []
        
        for head, relation, tail in triples:
            rel = {
                'source_id': head,
                'target_id': tail,
                'type': self._normalize_relation_type(relation),
                'original_relation': relation,
                'readable_name': self.resolver.resolve_relation(relation)
            }
            relationships.append(rel)
        
        return relationships
    
    @staticmethod
    def _infer_entity_type(entity_id: str) -> str:
        """Infer entity type from ID pattern"""
        if entity_id.startswith('/m/'):
            return 'MID'  # Machine ID
        elif entity_id.startswith('/g/'):
            return 'GID'  # Google ID
        else:
            return 'Unknown'
    
    @staticmethod
    def _normalize_relation_type(relation: str) -> str:
        """Normalize relation for Neo4j (alphanumeric + underscore)"""
        # Remove leading slashes and convert to uppercase with underscores
        normalized = relation.strip('/')
        normalized = normalized.replace('/', '_').replace('.', '_')
        normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
        return normalized.upper()