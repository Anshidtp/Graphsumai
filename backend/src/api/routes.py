from fastapi import APIRouter, HTTPException
from typing import List
import logging

from src.api.models import (
    QueryRequest, QueryResponse,
    EntitySearchRequest, EntityResponse,
    NeighborsRequest, GraphStatsResponse
)
from src.k_graph.graph_builder import GraphConstructor
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances
query_engine = None
retriever = None
generator = None


def set_dependencies(qe, ret, gen):
    """Set global dependencies"""
    global query_engine, retriever, generator
    query_engine = qe
    retriever = ret
    generator = gen


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest):
    """Query the knowledge graph with natural language"""
    try:
        logger.info(f"📥 Received query: {request.query}")
        
        # Retrieve context
        retrieval_result = retriever.retrieve(
            request.query,
            top_k=request.top_k,
            depth=request.depth
        )
        
        # Generate answer
        generation_result = generator.generate(
            request.query,
            retrieval_result['context']
        )
        
        return QueryResponse(
            query=request.query,
            answer=generation_result['answer'],
            context=retrieval_result['context'],
            metadata=retrieval_result['metadata'],
            status=generation_result['status']
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[EntityResponse])
async def search_entities(request: EntitySearchRequest):
    """Search for entities by name or ID"""
    try:
        results = query_engine.search_entities(
            request.search_term,
            limit=request.limit
        )
        
        return [
            EntityResponse(
                id=e['id'],
                name=e.get('name', ''),
                description=e.get('description'),
                degree=e.get('degree', 0),
                aliases=e.get('aliases', [])
            )
            for e in results
        ]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neighbors")
async def get_neighbors(request: NeighborsRequest):
    """Get neighbors of an entity"""
    try:
        neighbors = query_engine.get_neighbors(
            request.entity_id,
            limit=request.limit
        )
        
        return {
            'entity_id': request.entity_id,
            'neighbors': neighbors
        }
    
    except Exception as e:
        logger.error(f"Neighbors error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=GraphStatsResponse)
async def get_stats():
    """Get graph statistics"""
    try:
        
        constructor = GraphConstructor(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD
        )
        
        stats = constructor.get_statistics()
        constructor.close()
        
        return GraphStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "query_engine": query_engine is not None,
            "retriever": retriever is not None,
            "generator": generator is not None
        }
    }