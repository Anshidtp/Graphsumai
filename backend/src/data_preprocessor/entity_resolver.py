import re
import requests
from typing import Dict, Optional, List
from functools import lru_cache
import json
import time
import logging

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Enhanced entity resolver that:
    1. Fetches REAL names from Wikidata API
    2. Creates searchable aliases
    3. Caches results to avoid repeated API calls
    """
    
    def __init__(self, cache_file: str = "data/entity_cache.json"):
        self.cache_file = cache_file
        self.entity_cache: Dict[str, str] = {}
        self.relation_cache: Dict[str, str] = {}
        self.load_cache()
        
    def load_cache(self):
        """Load cached entity names from file"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.entity_cache = data.get('entities', {})
                self.relation_cache = data.get('relations', {})
            logger.info(f"✅ Loaded {len(self.entity_cache)} cached entities")
        except FileNotFoundError:
            logger.warning("⚠️  No cache file found, will create new one")
            
    def save_cache(self):
        """Save entity names to cache file"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'entities': self.entity_cache,
                'relations': self.relation_cache
            }, f, indent=2)
        logger.info(f"💾 Saved {len(self.entity_cache)} entities to cache")
    
    @lru_cache(maxsize=50000)
    def resolve_entity(self, freebase_id: str) -> str:
        """
        Resolve Freebase ID to human-readable name
        
        FIXED APPROACH:
        1. Check cache first
        2. Try Wikidata API (Freebase → Wikidata → Label)
        3. Fall back to cleaned ID if API fails
        """
        
        # Return cached if available
        if freebase_id in self.entity_cache:
            return self.entity_cache[freebase_id]
        
        # Try to fetch from Wikidata
        name = self._fetch_from_wikidata(freebase_id)
        
        if name:
            self.entity_cache[freebase_id] = name
            return name
        
        # Fallback: Clean the Freebase ID
        cleaned = self._clean_freebase_id(freebase_id)
        self.entity_cache[freebase_id] = cleaned
        return cleaned
    
    def _fetch_from_wikidata(self, freebase_id: str) -> Optional[str]:
        """
        Fetch entity name from Wikidata API
        
        Process:
        /m/0grwj (Freebase) → Q9798 (Wikidata) → "Jackie Chan" (Label)
        """
        try:
            # Step 1: Get Wikidata ID from Freebase ID
            sparql_query = f"""
            SELECT ?item ?itemLabel WHERE {{
              ?item wdt:P646 "{freebase_id}" .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 1
            """
            
            url = "https://query.wikidata.org/sparql"
            headers = {'User-Agent': 'GraphRAG/1.0'}
            params = {'query': sparql_query, 'format': 'json'}
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                
                if bindings:
                    label = bindings[0].get('itemLabel', {}).get('value')
                    if label:
                        logger.debug(f"✅ Resolved {freebase_id} → {label}")
                        return label
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.debug(f"⚠️  Wikidata lookup failed for {freebase_id}: {e}")
        
        return None
    
    def _clean_freebase_id(self, freebase_id: str) -> str:
        """
        Clean Freebase ID as fallback
        /m/0grwj → M0grwj
        """
        cleaned = freebase_id.replace('/', '').replace('.', '_')
        return cleaned.title()
    
    def resolve_relation(self, relation: str) -> str:
        """
        Clean relation name
        /people/person/profession → Person_Profession
        """
        if relation in self.relation_cache:
            return self.relation_cache[relation]
        
        # Remove namespace
        parts = relation.split('/')
        if len(parts) > 2:
            # Get last meaningful parts
            clean_parts = [p for p in parts if p and p not in ['people', 'film', 'location']]
            cleaned = '_'.join(clean_parts[-2:]) if len(clean_parts) >= 2 else clean_parts[-1]
        else:
            cleaned = relation.replace('/', '_')
        
        # Convert to Title Case
        cleaned = cleaned.replace('_', ' ').title().replace(' ', '_')
        
        self.relation_cache[relation] = cleaned
        return cleaned
    
    def batch_resolve_entities(self, freebase_ids: List[str], batch_size: int = 50):
        """
        Resolve multiple entities in batches with progress tracking
        """
        total = len(freebase_ids)
        logger.info(f"🔄 Resolving {total} entities...")
        
        from tqdm import tqdm
        
        for i in tqdm(range(0, total, batch_size), desc="Resolving entities"):
            batch = freebase_ids[i:i+batch_size]
            for fid in batch:
                self.resolve_entity(fid)
            
            # Save cache periodically
            if (i + batch_size) % 500 == 0:
                self.save_cache()
        
        self.save_cache()
        logger.info("✅ Batch resolution complete!")
    
    def get_aliases(self, name: str) -> List[str]:
        """
        Generate searchable aliases for entity names
        
        Example:
        "Jackie Chan" → ["jackie chan", "chan", "jackie", "chan, jackie"]
        """
        aliases = [name.lower()]
        
        # Split into parts
        parts = name.lower().split()
        
        # Add each part
        aliases.extend(parts)
        
        # Add last name first format
        if len(parts) >= 2:
            aliases.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
        
        # Remove duplicates
        return list(set(aliases))