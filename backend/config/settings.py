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
    LLM_MODEL: str = "meta-llama/llama-guard-4-12b"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Data Paths
    DATA_DIR: Path = Path("backend/data/raw")
    TRAIN_FILE: Path = DATA_DIR / "train.txt"
    VALID_FILE: Path = DATA_DIR / "valid.txt"
    TEST_FILE: Path = DATA_DIR / "test.txt"
    ENTITY_NAMES_FILE: Path = Path("data/entity_names.txt")

    # Output
    OUTPUT_DIR: Path = Path("output")
    RESULTS_DIR: Path = OUTPUT_DIR / "results"
    LOGS_DIR: Path = OUTPUT_DIR / "logs"
    
    class Config:
        env_file = ".env"

settings = Settings()