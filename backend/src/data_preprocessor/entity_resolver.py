import re
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Resolve Freebase MIDs to human-readable names
    
    FB15k-237 uses Freebase MIDs like /m/0grwj
    This class helps convert them to readable names
    """
    
    def __init__(self, entity_names_file: Optional[Path] = None):
        self.entity_to_name: Dict[str, str] = {}
        
        if entity_names_file and entity_names_file.exists():
            self._load_entity_names(entity_names_file)
    
    def _load_entity_names(self, filepath: Path):
        """Load entity ID to name mapping"""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id, name = parts[0], parts[1]
                    self.entity_to_name[entity_id] = name
        
        logger.info(f"Loaded {len(self.entity_to_name)} entity names")
    
    def resolve_entity(self, entity_id: str) -> str:
        """Convert entity ID to readable name"""
        if entity_id in self.entity_to_name:
            return self.entity_to_name[entity_id]
        
        # Fallback: clean up the ID
        return self._clean_entity_id(entity_id)
    
    def resolve_relation(self, relation: str) -> str:
        """Convert relation to readable name"""
        # Remove namespace prefix
        relation = relation.split('/')[-1]
        
        # Convert underscores to spaces and capitalize
        relation = relation.replace('_', ' ').title()
        
        return relation
    
    @staticmethod
    def _clean_entity_id(entity_id: str) -> str:
        """Clean entity ID for display"""
        # Remove namespace prefix /m/ or /g/
        cleaned = re.sub(r'^/[mg]/', '', entity_id)
        return f"Entity_{cleaned}"
    
    def format_triple(self, head: str, relation: str, tail: str) -> str:
        """Format triple with resolved names"""
        return f"{self.resolve_entity(head)} - {self.resolve_relation(relation)} - {self.resolve_entity(tail)}"
