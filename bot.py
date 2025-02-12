#!/usr/bin/env python3
"""
Telegram Business Coder Bot using LangChain and MongoDB.

This bot leverages GPT-based models to provide business advice via Telegram.
It supports multimodal inputs (text and images) and uses persistent history storage.
It also includes history optimization via summarization.

LangChain History & Memory Options:
    - InMemoryChatMessageHistory: Ephemeral, stored in RAM.
    - MongoDBChatMessageHistory: Persistent history stored in MongoDB.
    - ConversationBufferMemory: Stores entire conversation.
    - ConversationBufferWindowMemory: Stores only a recent window of messages.
    - ConversationSummaryMemory: Summarizes older conversation for efficiency.
    - ConversationEntityMemory: Tracks and extracts key entities.
    
This code uses:
    • MongoDBChatMessageHistory for persistence.
    • A summarization chain (via load_summarize_chain) to generate summaries.
    
The bot has the following new commands:
    - /new_chat: Create a new conversation session.
    - /history: List all previous sessions (with counts and summaries) for the current chat.
    - /summarize: Show a summary of the current conversation session.

The code is organized into:
    1. LangChain Integration (LLM initialization, prompt template, history chain)
    2. Telegram Bot Integration (command handlers, multimodal input handling)
    
Multimodal Input Instructions:
    For photo messages, we prepare a list of blocks (text and image block with a data URI)
    and use a helper function to format that list into a string for prompt logging.
"""

# ============================================
# Imports and Global Setup
# ============================================
import os
import datetime
import logging
import threading
import atexit
import asyncio
import time
import json
import base64
import requests
import telebot
from pymongo import MongoClient

# For Windows: use WindowsSelectorEventLoopPolicy to avoid warnings
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Disable SSL verification for Telebot API helper (TESTING ONLY)
import telebot.apihelper as apihelper
apihelper.SESSION = requests.Session()
apihelper.SESSION.verify = False  # WARNING: Disable SSL verification ONLY for testing

from telebot.types import BotCommandScopeAllGroupChats, BotCommandScopeDefault, BotCommand

# ============================================
# Global Configuration and API Keys
# ============================================
TELEGRAM_BOT_TOKEN = "7796762427:AAGDTTAt6qn0-bTpnkejqsy8afQJLZhWkuk"  # Replace with your actual token
GOOGLE_API_KEY = "AIzaSyBAHu5yR3ooMkyVyBmdFxw-8lWyaExLjjE"           # Replace with your actual API key
OPENAI_API_KEY = "123"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global dictionaries for per-chat settings
chat_session_map = {}     # Maps Telegram chat id to current session id (format: "{chat_id}_{timestamp}")
business_info_map = {}    # Stores business info per chat_id
ai_tone_map = {}          # Stores AI tone per chat_id (default: "دوستانه")

# MongoDB configuration for chat histories
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "chat_db"
COLLECTION_NAME = "chat_histories"

# ============================================
# Logging Configuration
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Suppress excessive logs from external libraries.
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("g4f").setLevel(logging.INFO)

logging.info("Logging configured (INFO level for our messages, external libraries suppressed).")

# ============================================
# Start G4F Interference API Server
# ============================================
try:
    from g4f.api import run_api
except ImportError:
    logging.error("g4f.api module not found. Please install the 'g4f' package.")
    run_api = None

if run_api is not None:
    def start_interference_api():
        logging.info("Starting G4F Interference API server on http://localhost:15203/v1 ...")
        run_api(bind="0.0.0.0:15203")
    api_thread = threading.Thread(target=start_interference_api, daemon=True, name="G4F-API-Thread")
    api_thread.start()
else:
    logging.warning("G4F API server not started due to missing module.")

def wait_for_api_server(timeout=30):
    """
    Ping the /chat/completions endpoint until the server responds or a timeout occurs.
    """
    base_url = "http://localhost:15203/v1/chat/completions"
    start_time = datetime.datetime.now()
    logging.info("Waiting for the G4F API server to become available...")
    while True:
        try:
            response = requests.post(base_url, json={"messages": [{"role": "system", "content": "ping"}]}, timeout=5)
            if response.ok:
                logging.info("G4F API server is up and running.")
                return
        except Exception:
            pass  # Suppress repeated error messages.
        if (datetime.datetime.now() - start_time).seconds > timeout:
            logging.error("API server not available after waiting 30 seconds.")
            return
        time.sleep(1)

wait_for_api_server()

# ============================================
# Helper: Format Multimodal Input for Logging
# ============================================
def format_multimodal_input(input_val):
    """
    Convert a multimodal input (list of blocks) into a string for logging/prompt formatting.
    """
    if isinstance(input_val, list):
        parts = []
        for block in input_val:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                url = block.get("image_url", {}).get("url", "")
                parts.append(f"[IMAGE: {url[:50]}...]")
        return "\n".join(parts)
    return str(input_val)

# ============================================
# SECTION 1: LangChain Integration
# ============================================
logging.info("Initializing LangChain LLM integrations...")

# Import LangChain modules and message types
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Initialize primary LangChain LLM for business advice
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gemini-1.5-flash",
    temperature=0.5
)
logging.info("Primary LangChain LLM initialized.")

# Initialize secondary LangChain LLM for business info summarization (if needed)
llm_business = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gemini-1.5-flash",
    temperature=0.5
)
logging.info("Secondary LangChain LLM for business info summarization initialized.")

atexit.register(lambda: llm.client.close() if hasattr(llm, "client") and callable(getattr(llm.client, "close", None)) else None)
atexit.register(lambda: llm_business.client.close() if hasattr(llm_business, "client") and callable(getattr(llm_business.client, "close", None)) else None)

# Define the prompt template using chat message templates.
prompt_template_text = """
You are Blue, a professional AI business coach using the GPT-4o model. Your role is to provide concise, actionable, and personalized business advice based on user input and session history.

**Response Requirements:**
- Answer in Markdown using clear headings and bullet points.
- Keep your response brief (no more than 150 words) and include only essential information.
- Use the user's name in every response for a personal touch.

**Guidelines:**
- Follow the user's request exactly; do not include extra background details unless asked.
- Maintain your tone as `{ai_tone}`.
- Reference the business context `{business_info}` only when directly relevant.

**Multimodal Input Note:**
If the input is a list containing a block with "type": "image_url", the "url" field holds a data URI of an image.
Please analyze the image content accordingly.

Important: Answer all messages in Persian.
"""
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template_text),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
logging.info("LangChain prompt template created.")

# Create a chain with message history using MongoDB
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda telegram_chat_id: get_history_for_chat(telegram_chat_id),
    input_messages_key="input",
    history_messages_key="history",
)
logging.info("LangChain chain with message history configured.")

# --------------------------------------------
# History Summarization Helper
# --------------------------------------------
from langchain.chains.summarize import load_summarize_chain

def get_summarized_history_for_session(session_id: str) -> str:
    """
    Load messages from a session (using MongoDBChatMessageHistory) and generate a summary.
    """
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    messages = history_obj.messages
    if not messages:
        return "No messages."
    combined = "\n".join([msg.content if hasattr(msg, "content") else str(msg) for msg in messages])
    try:
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = summary_chain.run(combined)
    except Exception as e:
        summary = f"Summary failed: {str(e)}"
    return summary

# ============================================
# MongoDB Helper Functions
# ============================================
def get_mongo_collection():
    """
    Connect to MongoDB and return the chat histories collection.
    """
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    logging.info("Connected to MongoDB (database: '%s').", DATABASE_NAME)
    return db[COLLECTION_NAME]

def get_history_for_chat(telegram_chat_id: str):
    """
    Retrieve or create the chat history object for a given Telegram chat.
    """
    if telegram_chat_id not in chat_session_map:
        new_session_id = f"{telegram_chat_id}_{int(datetime.datetime.now().timestamp())}"
        chat_session_map[telegram_chat_id] = new_session_id
        logging.info("Created new session '%s' for chat '%s'.", new_session_id, telegram_chat_id)
    session_id = chat_session_map[telegram_chat_id]
    logging.debug("Retrieving history for session '%s' in chat '%s'.", session_id, telegram_chat_id)
    
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    try:
        messages = history_obj.messages
        logging.info("Loaded %d historical messages for session '%s'.", len(messages), session_id)
    except Exception as e:
        logging.warning("Could not retrieve history messages for session '%s': %s", session_id, e)
    return history_obj

def get_history_count(session_id: str):
    """
    Return the count of history messages for a given session.
    """
    collection = get_mongo_collection()
    count = collection.count_documents({"session_id": {"$regex": f"^{session_id}"}})
    logging.debug("Session '%s' has %d messages in history.", session_id, count)
    return count

def save_message_to_history(chat_id, role, content):
    """
    Save a message (with role: user, assistant, or system) to MongoDB history.
    Uses LangChain's built-in message types for proper serialization.
    """
    try:
        history_obj = get_history_for_chat(chat_id)
        if role == "user":
            message_obj = HumanMessage(content=content)
        elif role == "assistant":
            message_obj = AIMessage(content=content)
        elif role == "system":
            message_obj = SystemMessage(content=content)
        else:
            message_obj = HumanMessage(content=content)
        history_obj.add_message(message_obj)
        if isinstance(content, list):
            truncated = format_multimodal_input(content)
        else:
            truncated = content
        truncated = truncated[:50] + ("..." if len(truncated) > 50 else "")
        logging.info("Saved message to history for chat '%s'. Role: '%s', Content: '%s'", chat_id, role, truncated)
    except Exception as e:
        logging.error("Error saving message to history for chat '%s': %s", chat_id, e)

# ============================================
# SECTION 2: Telegram Bot Integration
# ============================================
logging.info("Initializing Telegram Bot...")

# Initialize the Telegram Bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
logging.info("Telegram Bot initialized successfully.")

def is_admin(chat_id, user_id):
    """
    Check if a user is an administrator in the given chat.
    """
    try:
        member = bot.get_chat_member(chat_id, user_id)
        is_admin_status = member.status in ['creator', 'administrator']
        logging.debug("Admin check for user '%s' in chat '%s': %s", user_id, chat_id, is_admin_status)
        return is_admin_status
    except Exception as e:
        logging.error("Error checking admin status for user '%s' in chat '%s': %s", user_id, chat_id, e)
        return False

def setup_bot_commands():
    """
    Set up the bot commands for both private and group chats.
    """
    commands = [
        BotCommand("start", "شروع ربات و نمایش اطلاعات چت"),
        BotCommand("options", "انتخاب گزینه‌ها"),
        BotCommand("new_chat", "ایجاد جلسه چت جدید"),
        BotCommand("history", "نمایش تاریخچه جلسات"),
        BotCommand("summarize", "خلاصه تاریخچه جلسه فعلی"),
        BotCommand("help", "نمایش پیام راهنما"),
        BotCommand("settings", "تنظیمات ربات"),
        BotCommand("about", "اطلاعات ربات")
    ]
    logging.info("Setting up bot commands...")
    try:
        bot.delete_my_commands(scope=BotCommandScopeDefault())
        bot.delete_my_commands(scope=BotCommandScopeAllGroupChats())
    except Exception as e:
        logging.error("Error deleting existing bot commands: %s", e)
    try:
        bot.set_my_commands(commands, scope=BotCommandScopeDefault())
        bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())
        logging.info("Bot commands set successfully.")
    except Exception as e:
        logging.error("Exception during bot command setup: %s", e)

# ---------------------------------------------
# New Chat Command: /new_chat
# ---------------------------------------------
@bot.message_handler(commands=['new_chat'])
def new_chat(message):
    """
    Create a new chat session by generating a new session ID.
    """
    chat_id = str(message.chat.id)
    new_session_id = f"{chat_id}_{int(datetime.datetime.now().timestamp())}"
    chat_session_map[chat_id] = new_session_id
    logging.info("New chat session created for chat '%s': %s", chat_id, new_session_id)
    response_text = "🆕 جلسه چت جدید ایجاد شد. اکنون شما یک گفتگو جدید دارید."
    bot.reply_to(message, response_text, parse_mode="Markdown")

# ---------------------------------------------
# History Command: /history
# ---------------------------------------------
@bot.message_handler(commands=['history'])
def show_history(message):
    """
    List all previous chat sessions for the current chat.
    For each session, show its session ID, creation time, message count, and a summary.
    """
    chat_id = str(message.chat.id)
    collection = get_mongo_collection()
    sessions = collection.distinct("session_id", {"session_id": {"$regex": f"^{chat_id}_"}})
    if not sessions:
        response_text = "هیچ تاریخچه چتی برای شما پیدا نشد."
    else:
        response_lines = []
        for s in sessions:
            from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
            history_obj = MongoDBChatMessageHistory(
                session_id=s,
                connection_string=MONGO_CONNECTION_STRING,
                database_name=DATABASE_NAME,
                collection_name=COLLECTION_NAME,
            )
            count = len(history_obj.messages)
            try:
                timestamp = int(s.split("_")[1])
                time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = "Unknown time"
            summary = get_summarized_history_for_session(s)
            response_lines.append(f"Session: `{s}`\nCreated: {time_str}\nMessages: {count}\nSummary: {summary}\n")
        response_text = "\n".join(response_lines)
    logging.info("History command executed for chat '%s'.", chat_id)
    bot.reply_to(message, response_text, parse_mode="Markdown")

# ---------------------------------------------
# Summarize Command: /summarize
# ---------------------------------------------
@bot.message_handler(commands=['summarize'])
def summarize_history(message):
    """
    Show a summary of the current session's history.
    """
    chat_id = str(message.chat.id)
    current_session = chat_session_map.get(chat_id)
    if not current_session:
        response_text = "هیچ تاریخچه فعالی پیدا نشد."
    else:
        summary = get_summarized_history_for_session(current_session)
        response_text = f"خلاصه تاریخچه جلسه فعلی (`{current_session}`):\n\n{summary}"
    logging.info("Summarize command executed for chat '%s'.", chat_id)
    bot.reply_to(message, response_text, parse_mode="Markdown")

# ---------------------------------------------
# Telegram Command Handlers: /start, /help, /about, /settings
# ---------------------------------------------
@bot.message_handler(commands=['start'])
def send_welcome(message):
    """
    Handle the /start command to welcome the user.
    """
    chat_id = str(message.chat.id)
    welcome_message = (
        "سلام! من بلو هستم، همراه هوشمند کسب‌وکار تو! 🚀✨\n\n"
        "با استفاده از هوش مصنوعی GPT-4o و تحلیل دقیق، به عنوان مربی حرفه‌ای کسب‌وکار کنارت هستم.\n\n"
        "برای دریافت مشاوره شخصی‌سازی‌شده، ابتدا اطلاعات بیزینس و تیم خود را با دستور /settings ثبت کن.⚙️\n\n"
        "**توانایی‌های من:**\n"
        "🟢 برنامه‌ریزی وظایف تیم\n"
        "🟢 ارائه گزارش‌های روزانه (/options)\n"
        "🟢 راهنمایی استراتژیک برای پروژه‌ها\n"
        "🟢 پاسخ‌گویی هوشمند بر اساس داده‌های کسب‌وکار\n"
        "🟢 تطبیق لحن پاسخ‌ها بر اساس تنظیمات (/settings)\n\n"
        "برای تعامل راحت‌تر:\n"
        "🔹 منو صدا بزن ('بلو')\n"
        "🔹 منو تگ کن (@Blue)\n"
        "🔹 از گزینه‌های بات استفاده کن\n\n"
        "فقط کافیست صدام کنی. 😉\n"
    )
    logging.info("Processing /start command for chat '%s'.", chat_id)
    bot.reply_to(message, welcome_message, parse_mode="Markdown")
    save_message_to_history(chat_id, "system", welcome_message)

@bot.message_handler(commands=['help'])
def send_help(message):
    """
    Handle the /help command to provide usage instructions.
    """
    help_text = (
        "🤖 *دستورات ربات:*\n\n"
        " - `/start` - شروع ربات و نمایش اطلاعات چت\n"
        " - `/new_chat` - ایجاد جلسه چت جدید\n"
        " - `/history` - نمایش تاریخچه جلسات\n"
        " - `/summarize` - خلاصه تاریخچه جلسه فعلی\n"
        " - `/options` - انتخاب گزینه‌ها\n"
        " - `/settings` - تنظیمات ربات (تنظیم لحن و اطلاعات کسب‌وکار)\n"
        " - `/help` - نمایش پیام راهنما\n"
        " - `/about` - اطلاعات ربات\n\n"
        "در گروه‌ها، من تنها زمانی پاسخ می‌دهم که منشن شوم یا کلمه *بلو* در پیام وجود داشته باشد."
    )
    logging.info("Processing /help command.")
    bot.reply_to(message, help_text, parse_mode="Markdown")
    save_message_to_history(str(message.chat.id), "system", help_text)

@bot.message_handler(commands=['about'])
def about_bot(message):
    """
    Handle the /about command to provide information about the bot.
    """
    about_text = (
        "🤖 *درباره ربات:*\n\n"
        "من **بلو** هستم، مربی کسب‌وکار هوشمند با پشتیبانی از فناوری LangChain و مدل‌های OpenAI.\n"
        "برای اطلاعات بیشتر از دستور `/help` استفاده کنید."
    )
    logging.info("Processing /about command for chat '%s'.", message.chat.id)
    bot.reply_to(message, about_text, parse_mode="Markdown")
    save_message_to_history(str(message.chat.id), "system", about_text)

@bot.message_handler(commands=['settings'])
def bot_settings(message):
    """
    Handle the /settings command to provide configuration options.
    """
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_business_info = telebot.types.InlineKeyboardButton("بارگذاری اطلاعات بیزینس", callback_data="load_business_info")
    btn_ai_tone = telebot.types.InlineKeyboardButton("انتخاب لحن هوش مصنوعی", callback_data="ai_tone")
    keyboard.add(btn_business_info, btn_ai_tone)
    settings_text = (
        "⚙️ *تنظیمات ربات:*\n\n"
        "لطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
    )
    logging.info("Processing /settings command for chat '%s'.", chat_id)
    bot.reply_to(message, settings_text, reply_markup=keyboard, parse_mode="Markdown")
    save_message_to_history(chat_id, "system", settings_text)

# ---------------------------------------------
# Telegram Content Handlers for Text and Photo
# ---------------------------------------------
@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    """
    Process messages containing photos:
      - Download and encode the image.
      - Construct a multimodal input as a list of blocks.
      - Invoke the LangChain chain with the multimodal content.
    """
    chat_id = str(message.chat.id)
    sender_first_name = message.from_user.first_name or message.from_user.username
    logging.info("Received a photo message in chat '%s' from user '%s'.", chat_id, sender_first_name)
    
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        logging.info("Downloading image from URL: %s", file_url)
        response = requests.get(file_url)
        if response.status_code == 200:
            image_data = base64.b64encode(response.content).decode("utf-8")
            image_data_uri = f"data:image/jpeg;base64,{image_data}"
            logging.info("Image downloaded and encoded successfully for chat '%s'.", chat_id)
            
            multimodal_content = [
                {"type": "text", "text": message.caption if message.caption else "تصویر ارسال شده:"},
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]
            formatted_input = format_multimodal_input(multimodal_content)
            logging.info("Constructed multimodal input for photo message in chat '%s'.\nInput preview:\n%s", chat_id, formatted_input)
            
            save_message_to_history(chat_id, "user", multimodal_content)
            bot.send_chat_action(chat_id, 'typing')
            placeholder_message = bot.reply_to(message, "🤔 *در حال پردازش تصویر و متن...*", parse_mode="Markdown")
            
            try:
                ai_tone = ai_tone_map.get(chat_id, "دوستانه")
                business_info = business_info_map.get(chat_id, "")
                history = get_history_for_chat(chat_id)
                log_data = {
                    "Chat ID": chat_id,
                    "User": sender_first_name,
                    "AI Tone": ai_tone,
                    "Business Info": business_info,
                    "History Message Count": len(history.messages)
                }
                logging.info("Invoking LangChain chain for photo message in chat '%s'.\nPrompt details:\n%s",
                             chat_id, json.dumps(log_data, indent=4, ensure_ascii=False))
                
                prompt_input = format_multimodal_input(multimodal_content)
                full_prompt = prompt.format(input=prompt_input, ai_tone=ai_tone, business_info=business_info, history=history.messages)
                logging.info("Full prompt sent to AI:\n%s", full_prompt)
                
                ai_response = chain_with_history.invoke(
                    {"input": prompt_input, "ai_tone": ai_tone, "business_info": business_info},
                    config={"configurable": {"session_id": chat_id}}
                )
                logging.info("LangChain chain returned response for chat '%s'.", chat_id)
                save_message_to_history(chat_id, "assistant", ai_response.content)
                logging.info("Chat session '%s' now has %d messages in history.",
                             get_history_for_chat(chat_id).session_id,
                             len(get_history_for_chat(chat_id).messages))
                
                bot.edit_message_text(ai_response.content, chat_id=chat_id,
                                      message_id=placeholder_message.message_id,
                                      parse_mode="Markdown")
            except Exception as e:
                error_message = (
                    "❌ *متأسفم، مشکلی در پردازش تصویر و متن پیش آمد. لطفاً دوباره تلاش کنید.*\n\n"
                    f"*Error:* `{str(e)}`"
                )
                bot.edit_message_text(error_message, chat_id=chat_id,
                                      message_id=placeholder_message.message_id,
                                      parse_mode="Markdown")
                logging.error("Error invoking LangChain chain for photo in chat '%s': %s", chat_id, e)
        else:
            bot.reply_to(message, "❌ *خطا در دانلود تصویر. لطفاً دوباره تلاش کنید.*", parse_mode="Markdown")
            logging.error("Error downloading image file for chat '%s'. HTTP status code: %s", chat_id, response.status_code)
    except Exception as ex:
        bot.reply_to(message, "❌ *خطا در پردازش تصویر. لطفاً دوباره تلاش کنید.*", parse_mode="Markdown")
        logging.error("Exception in handle_photo for chat '%s': %s", chat_id, ex)

@bot.message_handler(func=lambda message: message.text is not None and not message.text.startswith("/"))
def handle_message(message):
    """
    Process non-command text messages.
    In group chats, respond only if the bot is mentioned or the keyword "بلو" is present.
    """
    chat_type = message.chat.type
    chat_id = str(message.chat.id)
    user_message_text = message.text
    sender_first_name = message.from_user.first_name or message.from_user.username
    logging.info("Received text message in chat '%s' from user '%s'.", chat_id, sender_first_name)
    
    if message.reply_to_message:
        replied_text = message.reply_to_message.text or ""
        modified_user_message = (
            f"*در پاسخ به:*\n{replied_text}\n\n"
            f"*اسم کاربر:* `{sender_first_name}`\n{user_message_text}"
        )
    else:
        modified_user_message = f"*اسم کاربر:* `{sender_first_name}`\n{user_message_text}"

    save_message_to_history(chat_id, "user", modified_user_message)
    
    bot_username = bot.get_me().username
    is_mentioned = (
        (message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id) or
        (user_message_text and (bot_username in user_message_text)) or
        (message.entities and any(
            entity.type == 'mention' and user_message_text[entity.offset:entity.offset + entity.length].lower() == f"@{bot_username.lower()}"
            for entity in message.entities
        )) or ("بلو" in user_message_text)
    )
    if chat_type in ['group', 'supergroup'] and not is_mentioned:
        logging.info("Message in group '%s' ignored (bot not mentioned).", chat_id)
        return

    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.reply_to(message, "🤔 *در حال پردازش...*", parse_mode="Markdown")
    
    try:
        ai_tone = ai_tone_map.get(chat_id, "دوستانه")
        business_info = business_info_map.get(chat_id, "")
        history = get_history_for_chat(chat_id)
        log_data = {
            "Chat ID": chat_id,
            "User": sender_first_name,
            "AI Tone": ai_tone,
            "Business Info": business_info,
            "History Message Count": len(history.messages)
        }
        logging.info("Invoking LangChain chain for text message in chat '%s'.\nPrompt details:\n%s",
                     chat_id, json.dumps(log_data, indent=4, ensure_ascii=False))
        
        prompt_input = modified_user_message
        full_prompt = prompt.format(input=prompt_input, ai_tone=ai_tone, business_info=business_info, history=history.messages)
        logging.info("Full prompt sent to AI:\n%s", full_prompt)
        
        ai_response = chain_with_history.invoke(
            {"input": prompt_input, "ai_tone": ai_tone, "business_info": business_info},
            config={"configurable": {"session_id": chat_id}}
        )
        logging.info("LangChain chain returned response for chat '%s'.", chat_id)
        save_message_to_history(chat_id, "assistant", ai_response.content)
        logging.info("Chat session '%s' now has %d messages in history.",
                     get_history_for_chat(chat_id).session_id,
                     len(get_history_for_chat(chat_id).messages))
        
        bot.edit_message_text(ai_response.content, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="Markdown")
    except Exception as e:
        error_message = (
            "❌ *متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.*\n\n"
            f"*Error:* `{str(e)}`"
        )
        bot.edit_message_text(error_message, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="Markdown")
        logging.error("Error invoking LangChain chain for text message in chat '%s': %s", chat_id, e)

# ============================================
# Main Function to Run the Bot
# ============================================
def main():
    """
    Set up bot commands and start polling for incoming messages.
    """
    logging.info("Setting up bot commands...")
    setup_bot_commands()
    logging.info("Telegram Bot is now running. Chat sessions are managed via LangChain's MongoDB memory.")
    bot.polling(none_stop=True)

if __name__ == "__main__":
    main()
