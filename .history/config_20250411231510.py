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