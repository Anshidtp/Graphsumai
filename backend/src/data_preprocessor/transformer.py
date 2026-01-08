# from typing import List, Tuple, Dict
# import numpy as np
# from .entity_resolver import EntityResolver

# class DataPreprocessor:
#     """Preprocess FB15k-237 data for graph construction"""
    
#     def __init__(self, resolver: EntityResolver):
#         self.resolver = resolver
    
#     def create_nodes(self, entities: set) -> List[Dict]:
#         """Create node records from entities"""
#         nodes = []
        
#         for entity_id in entities:
#             node = {
#                 'id': entity_id,
#                 'label': 'Entity',
#                 'name': self.resolver.resolve_entity(entity_id),
#                 'freebase_id': entity_id,
#                 'type': self._infer_entity_type(entity_id)
#             }
#             nodes.append(node)
        
#         return nodes
    
#     def create_relationships(self, triples: List[Tuple[str, str, str]]) -> List[Dict]:
#         """Create relationship records from triples"""
#         relationships = []
        
#         for head, relation, tail in triples:
#             rel = {
#                 'source_id': head,
#                 'target_id': tail,
#                 'type': self._normalize_relation_type(relation),
#                 'original_relation': relation,
#                 'readable_name': self.resolver.resolve_relation(relation)
#             }
#             relationships.append(rel)
        
#         return relationships
    
#     @staticmethod
#     def _infer_entity_type(entity_id: str) -> str:
#         """Infer entity type from ID pattern"""
#         if entity_id.startswith('/m/'):
#             return 'MID'  # Machine ID
#         elif entity_id.startswith('/g/'):
#             return 'GID'  # Google ID
#         else:
#             return 'Unknown'
    
#     @staticmethod
#     def _normalize_relation_type(relation: str) -> str:
#         """Normalize relation for Neo4j (alphanumeric + underscore)"""
#         # Remove leading slashes and convert to uppercase with underscores
#         normalized = relation.strip('/')
#         normalized = normalized.replace('/', '_').replace('.', '_')
#         normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
#         return normalized.upper()\

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and resolve FB15k-237 data"""
    
    def __init__(self, resolver):
        self.resolver = resolver
    
    def resolve_and_save_entities(self, entities: set, output_file: Path):
        """
        STEP 1: Resolve all entities and save to CSV
        
        Creates: entities_resolved.csv with columns [freebase_id, name, aliases]
        """
        logger.info(f"\n{'='*60}")
        logger.info("STEP 1: Resolving Entities with Wikidata")
        logger.info(f"{'='*60}\n")
        
        # Batch resolve all entities
        entity_list = list(entities)
        self.resolver.batch_resolve_entities(entity_list)
        
        # Prepare data
        resolved_data = []
        for entity_id in tqdm(entity_list, desc="Preparing entity data"):
            name = self.resolver.resolve_entity(entity_id)
            aliases = ','.join(self.resolver.get_aliases(name))
            
            resolved_data.append({
                'freebase_id': entity_id,
                'name': name,
                'aliases': aliases
            })
        
        # Save to CSV
        df = pd.DataFrame(resolved_data)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"\n✅ Saved {len(resolved_data):,} entities to {output_file}")
        
        return df
    
    def resolve_and_save_triplets(self, triples: List[Tuple[str, str, str]], 
                                   output_file: Path):
        """
        Resolve triples and save DEDUPLICATED readable triplets
        
        IMPROVEMENTS:
        1. Deduplicates based on triplet text
        2. Normalizes relation names
        3. Creates canonical triplet format
        """
        logger.info(f"\n{'='*60}")
        logger.info("STEP 1: Creating Deduplicated Readable Triplets")
        logger.info(f"{'='*60}\n")
        
        resolved_triplets = []
        seen_triplets = set()  
        
        for head, relation, tail in tqdm(triples, desc="Resolving triplets"):
            head_name = self.resolver.resolve_entity(head)
            tail_name = self.resolver.resolve_entity(tail)
            relation_clean = self.resolver.resolve_relation(relation)
            
            # Create canonical triplet (lowercase for dedup)
            canonical = f"{head_name.lower()}|{relation_clean.lower()}|{tail_name.lower()}"
            
            # Skip if duplicate
            if canonical in seen_triplets:
                continue
            
            seen_triplets.add(canonical)
            
            # Create readable triplet text
            triplet_text = f"{head_name} {relation_clean} {tail_name}"
            
            resolved_triplets.append({
                'triplet_text': triplet_text,
                'head_name': head_name,
                'relation': relation_clean,
                'tail_name': tail_name
            })
        
        # Save to CSV
        df = pd.DataFrame(resolved_triplets)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        duplicates_removed = len(triples) - len(resolved_triplets)
        
        logger.info(f"\n✅ Saved {len(resolved_triplets):,} unique triplets")
        logger.info(f"   Removed {duplicates_removed:,} duplicates")
        logger.info(f"\nExample triplets:")
        for i in range(min(5, len(resolved_triplets))):
            logger.info(f"  {resolved_triplets[i]['triplet_text']}")
        
        return df