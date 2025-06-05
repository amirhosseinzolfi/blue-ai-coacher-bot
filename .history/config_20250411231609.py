"""Configuration settings for the Blue Business Bot."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

############################################
# Bot Configuration
############################################
TELEGRAM_BOT_TOKEN = "7571693917:AAFt87IDOmrC-r2CFEpU_y0d04jSBIpUOfw"
GOOGLE_API_KEY = "AIzaSyBQUW4FE5yCJMvNsmBfhec4WQvPCVtxJyw"
OPENAI_API_KEY = "453"

############################################
# MongoDB Configuration
############################################
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017')
DATABASE_NAME = os.getenv('MONGO_DATABASE', 'blue_business_bot')
COLLECTION_NAME = 'chat_history'
BUSINESS_INFO_COLLECTION = 'business_info'

############################################
# State Management
############################################
# Chat session management
chat_session_map = {}  # Maps chat_id to session_id

# Business info management
business_info_map = {}  # Maps chat_id to business info
business_info_update_pending = {}  # Flags for pending business info updates
business_info_mode = {}  # Mode for business info updates (replace/append)

# AI tone management
ai_tone_map = {}  # Maps chat_id to AI tone preference
ai_tone_update_pending = {}  # Flags for pending AI tone updates

############################################
# State Management with Permanent Storage
############################################
# These maps now serve as temporary caches, backed by MongoDB
chat_session_map = {}  # Maps chat_id to session_id
business_info_map = {}  # Maps chat_id to business info
ai_tone_map = {}  # Maps chat_id to AI tone preference

# Initialize maps from permanent storage
def init_from_permanent_storage():
    from db_manager import db_manager
    try:
        # Load business info
        for doc in db_manager.business_info.find({}):
            if "chat_id" in doc:
                business_info_map[doc["chat_id"]] = doc.get("business_info", "")
        
        # Load AI tones
        for doc in db_manager.ai_tones.find({}):
            if "chat_id" in doc:
                ai_tone_map[doc["chat_id"]] = doc.get("tone", "دوستانه")
    except Exception as e:
        print(f"Error initializing from permanent storage: {e}")

# Initialize at module import
init_from_permanent_storage()

############################################
# Chat History Configuration
############################################
MAX_HISTORY_LENGTH = 50  # Maximum number of messages to keep in history
SUMMARIZATION_THRESHOLD = 10  # Number of messages before triggering summarization

############################################
# Session Management
############################################
SESSION_TIMEOUT = 3600  # Session timeout in seconds (1 hour)
CLEANUP_INTERVAL = 86400  # Cleanup interval in seconds (24 hours)

############################################
# Export Configuration
############################################
__all__ = [
    'TELEGRAM_BOT_TOKEN',
    'GOOGLE_API_KEY',
    'OPENAI_API_KEY',
    'MONGO_CONNECTION_STRING',
    'DATABASE_NAME',
    'COLLECTION_NAME',
    'BUSINESS_INFO_COLLECTION',
    'chat_session_map',
    'business_info_map',
    'business_info_update_pending',
    'business_info_mode',
    'ai_tone_map',
    'ai_tone_update_pending',
    'MAX_HISTORY_LENGTH',
    'SUMMARIZATION_THRESHOLD',
    'SESSION_TIMEOUT',
    'CLEANUP_INTERVAL'
]
# ... (Keep existing config variables) ...

# --- LangGraph Configuration ---
# Define the MongoDB collection name for LangGraph checkpoints
LANGGRAPH_CHECKPOINT_COLLECTION = "langgraph_checkpoints_v2" # Use a distinct name

# --- Potentially move LLM model names here ---
# PRIMARY_LLM_MODEL = "gpt-4o"
# BUSINESS_LLM_MODEL = "gemini-2.0-flash"
# USER_REPORT_LLM_MODEL = "gpt-4o"
# SUMMARY_LLM_MODEL = "gemini-2.0-flash"

# --- Remove obsolete in-memory maps ---
# chat_session_map = {} # REMOVED
# business_info_map = {} # REMOVED
# ai_tone_map = {} # REMOVED
# business_info_update_pending = {} # REMOVED
# business_info_mode = {} # REMOVED
# ai_tone_update_pending = {} # REMOVED

# ... (Keep other config variables like API keys, Mongo connection, DB name) ...
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/" # Ensure this is correct
DATABASE_NAME = "telegram_bot_db"
COLLECTION_NAME = "chat_history_legacy" # Maybe rename if old history is kept
BUSINESS_INFO_COLLECTION = "business_info"
# USER_INFO_COLLECTION = "user_info" # Defined within db_manager? Standardize access.