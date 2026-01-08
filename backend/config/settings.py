from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import os

class Settings(BaseSettings):
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # LLM Configuration
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Data Paths
    DATA_DIR: Path = Path("backend/data/raw")
    TRAIN_FILE: Path = DATA_DIR / "train.txt"
    VALID_FILE: Path = DATA_DIR / "valid.txt"
    TEST_FILE: Path = DATA_DIR / "test.txt"
    RESOLVED_DIR: Path = Path("backend/data/resolved")
    CACHE_FILE: Path = Path("backend/data/entity_cache.json")
    EMBEDDING_CACHE_DIR: Path = Path("backend/data/embedding_cache")

    # Output
    OUTPUT_DIR: Path = Path("output")
    RESULTS_DIR: Path = OUTPUT_DIR / "results"
    LOGS_DIR: Path = OUTPUT_DIR / "logs"

    BATCH_SIZE: int = 1000
    MAX_NEIGHBORS: int = 50
    SIMILARITY_THRESHOLD: float = 0.7
    WIKIDATA_BATCH_SIZE: int = 50
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create directories
for directory in [settings.DATA_DIR, settings.RESOLVED_DIR, settings.CACHE_FILE.parent, settings.EMBEDDING_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)