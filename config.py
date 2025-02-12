# Configuration file for Blue AI Coacher Bot

import os

# Global Configuration and API Keys
TELEGRAM_BOT_TOKEN = "7796762427:AAGDTTAt6qn0-bTpnkejqsy8afQJLZhWkuk"  # Replace with your actual token
GOOGLE_API_KEY = "AIzaSyBAHu5yR3ooMkyVyBmdFxw-8lWyaExLjjE"  # Replace with your actual key
OPENAI_API_KEY = "123"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global settings for chat sessions and other options
chat_session_map = {}            # Maps Telegram chat id to current session id
business_info_map = {}           # Business info per chat id
ai_tone_map = {}                 # AI tone per chat id (default: "دوستانه")
business_info_update_pending = {}  # Flag for pending business info update
business_info_mode = {}          # Tracks update mode ("replace" or "append")
ai_tone_update_pending = {}      # Pending update flag for AI tone

# MongoDB configuration
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "chat_db"
COLLECTION_NAME = "chat_histories"
BUSINESS_INFO_COLLECTION = "user_business_info"  # Collection for persistent business info
