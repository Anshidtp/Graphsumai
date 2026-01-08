from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for entities and relations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        logger.info(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache = {}
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_file = self.cache_dir / "embedding_cache.pkl"
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.cache)} cached embeddings")
        logger.info(f"✅ Embedding model loaded (dim={self.embedding_dim})")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text"""
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )   

        
        return embeddings
    
    def embed_triplet(self, triplet_text: str) -> np.ndarray:
        """
        Generate embedding for a single triplet sentence with caching
        
        Example: "Jackie Chan profession Actor" → [0.123, -0.456, ...]
        """
        if triplet_text in self.cache:
            return self.cache[triplet_text]
        
        embedding = self.model.encode(
            triplet_text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        self.cache[triplet_text] = embedding
        
        # Save cache if cache_dir is set
        if self.cache_dir:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        
        return embedding
    
    def batch_embed_triplets(self, triplet_texts: List[str], batch_size: int = 256) -> np.ndarray:
        """
        Batch embed multiple triplet sentences with caching
        """
        logger.info(f"🔄 Generating embeddings for {len(triplet_texts):,} triplets...")
        
        # Check cache first
        cached_embeddings = []
        to_compute = []
        indices = []
        
        for i, text in enumerate(triplet_texts):
            if text in self.cache:
                cached_embeddings.append((i, self.cache[text]))
            else:
                to_compute.append(text)
                indices.append(i)
        
        logger.info(f"📋 {len(cached_embeddings)} cached, {len(to_compute)} to compute")
        
        # Compute missing embeddings
        if to_compute:
            all_embeddings = []
            from tqdm import tqdm
            for i in tqdm(range(0, len(to_compute), batch_size), desc="Embedding triplets"):
                batch = to_compute[i:i+batch_size]
                embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=batch_size
                )
                all_embeddings.append(embeddings)
            
            computed_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            
            # Cache the new embeddings
            for text, emb in zip(to_compute, computed_embeddings):
                self.cache[text] = emb
        
        # Combine cached and computed in original order
        result = np.zeros((len(triplet_texts), self.embedding_dim))
        for i, emb in cached_embeddings:
            result[i] = emb
        if to_compute:
            for idx, emb in zip(indices, computed_embeddings):
                result[idx] = emb
        
        # Save cache
        if self.cache_dir:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"💾 Saved {len(self.cache)} embeddings to cache")
        
        logger.info(f"✅ Generated {len(result):,} embeddings")
        return result

