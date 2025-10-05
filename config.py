import os
from pathlib import Path

class Config:
    """Configuration settings for the production app"""

    # ChromaDB Configuration
    CHROMADB_API_KEY = os.getenv('CHROMADB_API_KEY', 'ck-Fy8amGLBg4KBPKkQYshx257pB4hBrw3ZBCSoVUoKiyWa')
    CHROMADB_TENANT = os.getenv('CHROMADB_TENANT', 'ee54a044-1346-4b7c-89b7-d7554fb9aacf')
    CHROMADB_DATABASE = os.getenv('CHROMADB_DATABASE', 'RAG-chiangmai-travel')

    # LLM Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY','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')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY','AIzaSyDmMUkyeXIU4CvcKhh47xp3Y-28be53CYE')
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
        required_vars = []

        if not cls.GOOGLE_API_KEY:
            required_vars.append('GOOGLE_API_KEY')

        if required_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")

    @classmethod
    def setup_directories(cls):
        """Setup required directories"""
        cls.SESSIONS_DIR.mkdir(exist_ok=True)
