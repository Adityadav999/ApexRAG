import sys
import os
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseModel
        class BaseSettings(BaseModel):
            pass
    except ImportError:
        class BaseSettings:
            pass

class Settings(BaseSettings):
    APP_NAME: str = "ApexRAG Enterprise Engine"
    ENV: str = "production"
    
    # Storage & Persistence
    CHROMA_PERSIST_DIR: str = os.path.join(os.getcwd(), "data", "chroma_db")
    BM25_PERSIST_PATH: str = os.path.join(os.getcwd(), "data", "bm25_index.pkl")
    DOCS_DIR: str = os.path.join(os.getcwd(), "data", "documents")
    
    # Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_PROVIDER: str = "openai" # "openai", "ollama", or "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = None
    
    # RAG Tuning Parameters
    TOP_K_SPARSE: int = 15
    TOP_K_DENSE: int = 15
    TOP_K_RRF: int = 10
    TOP_K_RERANK: int = 5
    RRF_K: int = 60
    RELEVANCE_THRESHOLD: float = 0.30
    MAX_SELF_CORRECTION_LOOPS: int = 2
    
    # Langfuse Telemetry
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    ENABLE_LANGFUSE: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
