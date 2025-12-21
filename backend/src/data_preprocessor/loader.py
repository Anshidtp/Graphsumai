import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and parse FB15k-237 dataset
    
    Format: Tab-separated triples (head, relation, tail)
    Example: /m/0grwj	/people/person/profession	/m/05sxg2
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / "train.txt"
        self.valid_file = self.data_dir / "valid.txt"
        self.test_file = self.data_dir / "test.txt"
        
        self.entities = set()
        self.relations = set()
        self.triples = {'train': [], 'valid': [], 'test': []}
        
    def load_split(self, filepath: Path, split_name: str) -> List[Tuple[str, str, str]]:
        """
        Load a single split file
        
        Returns: List of (head, relation, tail) tuples
        """
        triples = []
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return triples
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 3:
                    logger.warning(f"Invalid line {line_num} in {filepath}: {line}")
                    continue
                
                head, relation, tail = parts
                triples.append((head, relation, tail))
                
                # Collect unique entities and relations
                self.entities.add(head)
                self.entities.add(tail)
                self.relations.add(relation)
        
        logger.info(f"Loaded {len(triples)} triples from {split_name}")
        return triples
    
    def load_all_splits(self) -> Dict[str, List[Tuple]]:
        """Load all dataset splits"""
        logger.info("Loading FB15k-237 dataset...")
        
        self.triples['train'] = self.load_split(self.train_file, 'train')
        self.triples['valid'] = self.load_split(self.valid_file, 'valid')
        self.triples['test'] = self.load_split(self.test_file, 'test')
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Entities: {len(self.entities)}")
        logger.info(f"  Relations: {len(self.relations)}")
        logger.info(f"  Train triples: {len(self.triples['train'])}")
        logger.info(f"  Valid triples: {len(self.triples['valid'])}")
        logger.info(f"  Test triples: {len(self.triples['test'])}")
        
        return self.triples
    
    def get_entity_statistics(self) -> Dict:
        """Get statistics about entities"""
        entity_counts = defaultdict(int)
        
        for split_triples in self.triples.values():
            for head, rel, tail in split_triples:
                entity_counts[head] += 1
                entity_counts[tail] += 1
        
        return {
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'avg_entity_frequency': sum(entity_counts.values()) / len(entity_counts),
            'max_entity_frequency': max(entity_counts.values()),
            'entity_counts': entity_counts
        }
    
    def get_relation_statistics(self) -> Dict:
        """Get statistics about relations"""
        relation_counts = defaultdict(int)
        
        for split_triples in self.triples.values():
            for head, rel, tail in split_triples:
                relation_counts[rel] += 1
        
        return {
            'relation_counts': relation_counts,
            'most_common_relations': sorted(
                relation_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def export_to_csv(self, output_dir: Path):
        """Export triples to CSV for easier inspection"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, triples in self.triples.items():
            df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
            output_path = output_dir / f"{split_name}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {split_name} to {output_path}")