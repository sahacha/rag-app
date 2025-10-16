import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

class Config:
    """Configuration settings for the production app"""

    # ChromaDB Configuration (secrets moved to .env)
    CHROMADB_API_KEY = os.getenv('CHROMADB_API_KEY')
    CHROMADB_TENANT = os.getenv('CHROMADB_TENANT')
    CHROMADB_DATABASE = os.getenv('CHROMADB_DATABASE', 'RAG-chiangmai-travel')

    # LLM Configuration (secrets moved to .env)
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    LLM_MODEL = os.getenv('LLM_MODEL', 'models/gemini-flash-latest')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '1'))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '4096'))
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '30'))

    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/distiluse-base-multilingual-cased')

    # Collection Configuration
    COLLECTIONS = ['all_collections_thai']
    RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '3'))

    # FlashRank Configuration
    FLASHRANK_MODEL = os.getenv('FLASHRANK_MODEL', 'ms-marco-MiniLM-L-12-v2')
    INITIAL_RETRIEVAL_K = int(os.getenv('INITIAL_RETRIEVAL_K', '10'))  # Documents to retrieve before reranking

    # Session Configuration
    SESSIONS_DIR = Path(os.getenv('SESSIONS_DIR', 'sessions'))

    # UI Configuration
    APP_TITLE = "ผู้ช่วยท่องเที่ยวเชียงใหม่"
    APP_ICON = "🏔️"
    MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '100'))

    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        # Make GOOGLE_API_KEY optional: we can run without Google search support.
        # If other required vars are added later, validate them here.
        return

    @classmethod
    def setup_directories(cls):
        """Setup required directories"""
        # We no longer create a sessions directory by default because history is
        # stored in browser localStorage. Leave directory creation to explicit
        # server-side deployments that still want it.
        return
