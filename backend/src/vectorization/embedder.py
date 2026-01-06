from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for entities and relations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
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
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch embed multiple texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

