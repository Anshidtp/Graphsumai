from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(10, description="Number of entities to retrieve", ge=1, le=50)


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    answer: str
    context: str
    metadata: Dict
    status: str


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics"""
    entities: int
    relationships: int
    avg_degree: float
    max_degree: int
    top_rel_types: List[Dict]
