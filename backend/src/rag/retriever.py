# from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict
import logging
import numpy as np
import sentence_transformers
import torch

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve context from knowledge graph"""
    
    def __init__(self, query_engine, model: str = "all-MiniLM-L6-v2",):
        self.query_engine = query_engine
        self.embeddings = sentence_transformers.SentenceTransformer(model=model)
    
    def retrieve_context(self, question: str, k: int = 10, depth: int = 2) -> str:
        """
        Retrieve relevant subgraph context for a question
        
        Steps:
        1. Extract entity mentions from question
        2. Find relevant entities in graph
        3. Expand to neighbors
        4. Format as context
        """
        logger.info(f"Retrieving context for: {question}")
        
        # Get relevant subgraph
        context = self.query_engine.get_subgraph_for_query(question, k)
        
        if not context or len(context) < 50:
            return "No relevant information found in the knowledge graph."
        
        return context
    
    def retrieve_for_link_prediction(self, head: str, relation: str) -> List[str]:
        """Retrieve context for link prediction task"""
        # Get neighbors of head entity
        neighbors = self.query_engine.get_entity_neighbors(head, depth=2)
        
        context_parts = []
        for item in neighbors[:20]:
            if 'neighbor' in item:
                neighbor = item['neighbor']
                context_parts.append(f"- {neighbor.get('name', neighbor.get('id'))}")
        
        return context_parts