from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # LLM Configuration
    LLM_API_KEY: str
    LLM_MODEL: str = "gpt-4"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Data Paths
    TRAIN_DATA_PATH: str = "data/raw/train"
    TEST_DATA_PATH: str = "data/raw/test"
    VAL_DATA_PATH: str = "data/raw/valid"
    
    class Config:
        env_file = ".env"

settings = Settings()