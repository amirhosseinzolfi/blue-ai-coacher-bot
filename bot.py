#!/usr/bin/env python3
"""
Telegram Business Coder Bot using LangChain and MongoDB.

This bot leverages GPT-based models to provide business advice via Telegram.
It supports multimodal inputs (text and images) and uses persistent history storage.
It is "group aware" – in group chats it records all messages (even those not addressed to the bot)
so that when a user asks a question the bot can answer with full context.

LangChain History & Memory Options:
    • InMemoryChatMessageHistory: Ephemeral, RAM-only.
    • MongoDBChatMessageHistory: Persistent history in MongoDB.
    • ConversationBufferMemory / WindowMemory: Full or sliding-window conversation.
    • ConversationSummaryMemory: Summarizes old conversation.
    • ConversationEntityMemory: Tracks key entities.

This bot uses MongoDBChatMessageHistory and a summarization chain to generate history summaries.

New Commands:
    - /new_chat: Start a new chat session.
    - /history: List all previous sessions (with counts and summaries) for this chat.
    - /options: Additional functions (Daily Tasks, Instagram Story Idea, Chat Report)

Multimodal Input Instructions:
    For photo messages, we construct a list of blocks (text and image_url blocks with a data URI).
    A helper function formats that list to a string for prompt logging.

All responses are in Persian.
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
import re  # For refining markdown responses and escaping characters
import telebot
from pymongo import MongoClient
import io
import PyPDF2  # Ensure PyPDF2 is installed
import g4f

from g4f.client import Client

client = Client()

# For Windows: avoid asyncio warnings
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Disable SSL verification for testing (DO NOT use in production)
import telebot.apihelper as apihelper
apihelper.SESSION = requests.Session()
apihelper.SESSION.verify = False

from telebot.types import BotCommandScopeAllGroupChats, BotCommandScopeDefault, BotCommand

# ============================================
# Global Configuration and API Keys
# ============================================
TELEGRAM_BOT_TOKEN = "7796762427:AAGDTTAt6qn0-bTpnkejqsy8afQJLZhWkuk"  # Replace with your actual token
GOOGLE_API_KEY = "AIzaSyBAHu5yR3ooMkyVyBmdFxw-8lWyaExLjjE"           # Replace with your actual key
OPENAI_API_KEY = "123"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global settings for chat sessions and other options
chat_session_map = {}     # Maps Telegram chat id to current session id
business_info_map = {}    # Business info per chat id
ai_tone_map = {}          # AI tone per chat id (default: "دوستانه")
business_info_update_pending = {}  # Flag for pending business info update
business_info_mode = {}   # NEW: Tracks update mode ("replace" or "append")
# NEW: Add pending update flag for AI tone
ai_tone_update_pending = {}

# MongoDB configuration
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "chat_db"
COLLECTION_NAME = "chat_histories"
BUSINESS_INFO_COLLECTION = "user_business_info"  # New collection for persistent business info

# ============================================
# Logging Configuration
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
    logging.error("g4f.api module not found. Install the 'g4f' package.")
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
    base_url = "http://localhost:15203/v1/chat/completions"
    start_time = datetime.datetime.now()
    logging.info("Waiting for the G4F API server to become available...")
    while True:
        try:
            response = requests.post(base_url, json={"messages": [{"role": "system", "content": "ping"}]}, timeout=5)
            if response.ok:
                logging.info("G4F API server responded successfully.")
                break  # Found server, break the loop.
        except Exception:
            pass
        if (datetime.datetime.now() - start_time).seconds > timeout:
            logging.error("API server not available after waiting 30 seconds.")
            return
        time.sleep(1)

wait_for_api_server()

# ============================================
# Helper: Format Multimodal Input for Logging
# ============================================
def format_multimodal_input(input_val):
    if isinstance(input_val, list):
        parts = []
        for block in input_val:
            if block.get("type") == "text":
                parts.append(block.get("content", ""))
            elif block.get("type") == "image_url":
                parts.append("[Image]")
        return "\n".join(parts)
    return str(input_val)

# ============================================
# Helper: Refine and Escape AI Markdown Response for Telegram View
# ============================================
def refine_ai_response(response_md: str) -> str:
    """
    Refines the AI's markdown response for a modern Telegram view by:
      1. Replacing headings with modern stickers
      2. Replacing list markers with 🔹
      3. Escaping special characters while preserving Markdown formatting
    """
    parts = response_md.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r'^####\s+(.*?)$', r'🔶 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^###\s+(.*?)$', r'⭐ \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^##\s+(.*?)$', r'🔷 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^#\s+(.*?)$', r'🟣 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*[-*]\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*\d+\.\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            temp = parts[i]
            placeholders = {
                'bold': (r'\*\*(.+?)\*\*', '‡B‡\\1‡B‡'),
                'italic': (r'\*(.+?)\*', '‡I‡\\1‡I‡'),
                'strike': (r'~~(.+?)~~', '‡S‡\\1‡S‡'),
                'code': (r'`(.+?)`', '‡C‡\\1‡C‡'),
                'link': (r'\[(.+?)\]\((.+?)\)', '‡L‡\\1‡U‡\\2‡L‡'),
            }
            for name, (pattern, repl) in placeholders.items():
                temp = re.sub(pattern, repl, temp)
            for char in special_chars:
                temp = temp.replace(char, f"\\{char}")
            restorations = {
                '‡B‡': '**',
                '‡I‡': '*',
                '‡S‡': '~~',
                '‡C‡': '`',
                '‡L‡': '[',
                '‡U‡': '](',
            }
            for placeholder, markdown in restorations.items():
                temp = temp.replace(placeholder, markdown)
            parts[i] = temp
        else:
            parts[i] = f'`{parts[i]}`'
    return ''.join(parts)

def escape_markdown_v2(text: str) -> str:
    """
    Escape special characters for Telegram's MarkdownV2 format.
    """
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    parts = text.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            for char in special_chars:
                parts[i] = parts[i].replace(char, f"\\{char}")
        else:
            parts[i] = f'`{parts[i]}`'
    return ''.join(parts)

# ============================================
# Group Message Listener (for Group Chats)
# ============================================
def group_message_listener(messages):
    for message in messages:
        if message.chat.type in ['group', 'supergroup']:
            chat_id = str(message.chat.id)
            sender = message.from_user.first_name or message.from_user.username
            if message.content_type != "text":
                if message.content_type == "photo":
                    text = f"{sender} sent a photo."
                    save_message_to_history(chat_id, "user", text)
                elif message.content_type in ['video', 'audio', 'document']:
                    text = f"{sender} sent a {message.content_type}."
                    save_message_to_history(chat_id, "user", text)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
bot.set_update_listener(group_message_listener)

# ============================================
# SECTION 1: LangChain Integration
# ============================================
logging.info("Initializing LangChain LLM integrations...")

from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gpt-4o",
    temperature=0.5
)
logging.info("Primary LangChain LLM initialized.")

llm_business = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gpt-4o",
    temperature=0.5
)
logging.info("Secondary LangChain LLM for business info summarization initialized.")

atexit.register(lambda: llm.client.close() if hasattr(llm, "client") and callable(getattr(llm.client, "close", None)) else None)
atexit.register(lambda: llm_business.client.close() if hasattr(llm_business, "client") and callable(getattr(llm_business.client, "close", None)) else None)

# ============================================
# Load Prompts from JSON file
# ============================================
with open(os.path.join(os.path.dirname(__file__), "prompts", "prompts.json"), "r", encoding="utf-8") as f:
    prompts = json.load(f)
prompt_template_text = prompts.get("prompt_template_text", "")
daily_task_prompt = prompts.get("daily_task_prompt", "")
summary_prompt = prompts.get("summary_prompt", "")
daily_report_prompt = prompts.get("daily_report_prompt", "")
insta_idea_prompt = prompts.get("insta_idea_prompt", "")
image_analyzer_prompt = prompts.get("image_analyzer_prompt", "")

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template_text),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
logging.info("LangChain prompt template created.")

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
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    logging.info("Connected to MongoDB (database: '%s').", DATABASE_NAME)
    return db[COLLECTION_NAME]

def get_business_info_collection():
    """Get MongoDB collection for storing permanent business info per user."""
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    return db[BUSINESS_INFO_COLLECTION]

def get_user_business_info(chat_id: str) -> str:
    """Get business info for a user from MongoDB."""
    collection = get_business_info_collection()
    result = collection.find_one({"chat_id": chat_id})
    return result["business_info"] if result else ""

def save_user_business_info(chat_id: str, info: str):
    """Save or update business info for a user in MongoDB."""
    collection = get_business_info_collection()
    collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"business_info": info, "updated_at": datetime.datetime.utcnow()}},
        upsert=True
    )
    logging.info(f"Updated business info for chat '{chat_id}'")

def get_history_for_chat(telegram_chat_id: str):
    if (telegram_chat_id not in chat_session_map):
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
    collection = get_mongo_collection()
    count = collection.count_documents({"session_id": {"$regex": f"^{session_id}"}})
    logging.debug("Session '%s' has %d messages in history.", session_id, count)
    return count

def save_message_to_history(chat_id, role, content):
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

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
logging.info("Telegram Bot initialized successfully.")

def is_admin(chat_id, user_id):
    try:
        member = bot.get_chat_member(chat_id, user_id)
        is_admin_status = member.status in ['creator', 'administrator']
        logging.debug("Admin check for user '%s' in chat '%s': %s", user_id, chat_id, is_admin_status)
        return is_admin_status
    except Exception as e:
        logging.error("Error checking admin status for user '%s' in chat '%s': %s", user_id, chat_id, e)
        return False

def setup_bot_commands():
    commands = [
        BotCommand("start", "شروع ربات و نمایش اطلاعات چت"),
        BotCommand("new_chat", "ایجاد جلسه چت جدید"),
        BotCommand("history", "نمایش تاریخچه جلسات"),
        BotCommand("options", "گزینه‌های اضافی"),
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

# -------------------------------
# /new_chat Command: Create a new session.
# -------------------------------
@bot.message_handler(commands=['new_chat'])
def new_chat(message):
    chat_id = str(message.chat.id)
    new_session_id = f"{chat_id}_{int(datetime.datetime.now().timestamp())}"
    chat_session_map[chat_id] = new_session_id
    logging.info("New chat session created for chat '%s': %s", chat_id, new_session_id)
    response_text = "🆕 جلسه چت جدید ایجاد شد. جلسه قبلی در تاریخچه ذخیره شده است."
    bot.reply_to(message, escape_markdown_v2(response_text), parse_mode="MarkdownV2")

# -------------------------------
# /history Command: List previous sessions.
# -------------------------------
@bot.message_handler(commands=['history'])
def show_history(message):
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
    bot.reply_to(message, response_text, parse_mode="MarkdownV2")

# -------------------------------
# /start Command: Send welcome message.
# -------------------------------
@bot.message_handler(commands=['start'])
def send_welcome(message):
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
    bot.reply_to(message, escape_markdown_v2(welcome_message), parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", welcome_message)

# -------------------------------
# /help Command: Send help information.
# -------------------------------
@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = (
        "🤖 *دستورات ربات:*\n\n"
        " - `/start` - شروع ربات و نمایش اطلاعات چت\n"
        " - `/new_chat` - ایجاد جلسه چت جدید\n"
        " - `/history` - نمایش تاریخچه جلسات\n"
        " - `/options` - گزینه‌های اضافی (وظایف روزانه، ایده استوری اینستاگرام، گزارش تاریخچه چت)\n"
        " - `/settings` - تنظیمات ربات (تنظیم لحن و اطلاعات کسب‌وکار)\n"
        " - `/help` - نمایش پیام راهنما\n"
        " - `/about` - اطلاعات ربات\n\n"
        "در گروه‌ها، من تنها زمانی پاسخ می‌دهم که منشن شوم یا کلمه *بلو* در پیام وجود داشته باشد."
    )
    logging.info("Processing /help command.")
    bot.reply_to(message, escape_markdown_v2(help_text), parse_mode="MarkdownV2")
    save_message_to_history(str(message.chat.id), "system", help_text)

# -------------------------------
# /about Command: Send information about the bot.
# -------------------------------
@bot.message_handler(commands=['about'])
def about_bot(message):
    about_text = (
        "🤖 *درباره ربات:*\n\n"
        "من **بلو** هستم، مربی کسب‌وکار هوشمند با پشتیبانی از فناوری LangChain و مدل‌های OpenAI.\n"
        "برای اطلاعات بیشتر از دستور `/help` استفاده کنید."
    )
    logging.info("Processing /about command for chat '%s'.", message.chat.id)
    bot.reply_to(message, escape_markdown_v2(about_text), parse_mode="MarkdownV2")
    save_message_to_history(str(message.chat.id), "system", about_text)

# -------------------------------
# /settings Command: Show settings options.
# -------------------------------
@bot.message_handler(commands=['settings'])
def bot_settings(message):
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
    bot.reply_to(message, settings_text, reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", settings_text)

# -------------------------------
# /options Command: Present additional functions.
# -------------------------------
@bot.message_handler(commands=['options'])
def options_handler(message):
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_daily_tasks = telebot.types.InlineKeyboardButton("تسک های روزانه", callback_data="daily_tasks")
    btn_instagram_story = telebot.types.InlineKeyboardButton("ایده استوری اینستاگرام", callback_data="instagram_story_idea")
    btn_chat_report = telebot.types.InlineKeyboardButton("گزارش روزانه", callback_data="chat_report")
    keyboard.add(btn_daily_tasks, btn_instagram_story, btn_chat_report)
    options_text = "⚙️ *انتخاب گزینه‌ها:*\nلطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
    logging.info("Processing /options command for chat '%s'.", chat_id)
    bot.reply_to(message, options_text, reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", options_text)

# -------------------------------
# Callback Handler for Daily Tasks Button
# -------------------------------
@bot.callback_query_handler(func=lambda call: call.data == "daily_tasks")
def handle_daily_tasks(call):
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logging.info("Daily Tasks option selected by '%s' in chat '%s'.", sender_first_name, chat_id)
    
    # Use JSON-based daily task prompt text
    prompt_input = daily_task_prompt
    
    ai_tone = ai_tone_map.get(chat_id, "دوستانه")
    business_info = business_info_map.get(chat_id, "")
    
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش وظایف روزانه..."), parse_mode="MarkdownV2")
    
    try:
        ai_response = chain_with_history.invoke(
            {"input": prompt_input, "ai_tone": ai_tone, "business_info": business_info},
            config={"configurable": {"session_id": chat_id}}
        )
        refined_response = refine_ai_response(ai_response.content)
        save_message_to_history(chat_id, "assistant", refined_response)
        logging.info("Daily Tasks response sent to chat '%s'.", chat_id)
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2(
            "❌ متأسفم، مشکلی در پردازش وظایف روزانه پیش آمد. لطفاً دوباره تلاش کنید."
        )
        logging.error("Error in Daily Tasks handler for chat '%s': %s", chat_id, e)
        bot.edit_message_text(
            error_message,
            chat_id=chat_id,
            message_id=placeholder_message.message_id,
            parse_mode="MarkdownV2"
        )

# -------------------------------
# Callback Handler for Instagram Story Idea Button
# -------------------------------
@bot.callback_query_handler(func=lambda call: call.data == "instagram_story_idea")
def handle_instagram_story_idea(call):
    chat_id = str(call.message.chat.id)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sender_first_name = call.from_user.first_name or call.from_user.username
    logging.info("Instagram Story Idea option selected by '%s' in chat '%s'.", sender_first_name, chat_id)
    
    business_info = business_info_map.get(chat_id, "")
    # Use JSON-based Instagram idea prompt text
    prompt_input = insta_idea_prompt
    
    ai_tone = ai_tone_map.get(chat_id, "دوستانه")
    
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش ایده استوری اینستاگرام..."), parse_mode="MarkdownV2")
    
    try:
        ai_response = chain_with_history.invoke(
            {"input": prompt_input, "ai_tone": ai_tone, "business_info": business_info},
            config={"configurable": {"session_id": chat_id}}
        )
        refined_response = refine_ai_response(ai_response.content)
        save_message_to_history(chat_id, "assistant", refined_response)
        logging.info("Instagram Story Idea response sent to chat '%s'.", chat_id)
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2(
            "❌ متأسفم، مشکلی در پردازش ایده استوری اینستاگرام پیش آمد. لطفاً دوباره تلاش کنید."
        )
        logging.error("Error in Instagram Story Idea handler for chat '%s': %s", chat_id, e)
        bot.edit_message_text(
            error_message,
            chat_id=chat_id,
            message_id=placeholder_message.message_id,
            parse_mode="MarkdownV2"
        )

# -------------------------------
# Callback Handler for Chat Report Button (Merged Summarize Functionality)
# -------------------------------
@bot.callback_query_handler(func=lambda call: call.data == "chat_report")
def handle_chat_report(call):
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logging.info("Chat Report option selected by '%s' in chat '%s'.", sender_first_name, chat_id)
    
    # Use JSON-based daily report prompt text
    prompt_input = daily_report_prompt
    
    ai_tone = ai_tone_map.get(chat_id, "دوستانه")
    business_info = business_info_map.get(chat_id, "")
    
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش گزارش تاریخچه چت کاربران..."), parse_mode="MarkdownV2")
    
    try:
        ai_response = chain_with_history.invoke(
            {"input": prompt_input, "ai_tone": ai_tone, "business_info": business_info},
            config={"configurable": {"session_id": chat_id}}
        )
        refined_response = refine_ai_response(ai_response.content)
        save_message_to_history(chat_id, "assistant", refined_response)
        logging.info("Chat Report response sent to chat '%s'.", chat_id)
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2(
            "❌ متأسفم، مشکلی در پردازش گزارش تاریخچه چت کاربران پیش آمد. لطفاً دوباره تلاش کنید."
        )
        logging.error("Error in Chat Report handler for chat '%s': %s", chat_id, e)
        bot.edit_message_text(
            error_message,
            chat_id=chat_id,
            message_id=placeholder_message.message_id,
            parse_mode="MarkdownV2"
        )

# -------------------------------
# New Callback Handler: Load Business Info
# -------------------------------
@bot.callback_query_handler(func=lambda call: call.data == "load_business_info")
def handle_load_business_info(call):
    chat_id = str(call.message.chat.id)
    current_info = get_user_business_info(chat_id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    if (current_info):
        # If business info exists, allow user to replace or append
        btn_replace = telebot.types.InlineKeyboardButton("تعویض اطلاعات", callback_data="replace_business_info")
        btn_append = telebot.types.InlineKeyboardButton("افزودن به اطلاعات", callback_data="append_business_info")
        keyboard.add(btn_replace, btn_append)
        prompt_text = f"اطلاعات بیزینس فعلی:\n{current_info}\n\nآیا می‌خواهید آن را تعویض کنید یا اطلاعات جدید به آن اضافه شود؟"
        bot.answer_callback_query(call.id)
        bot.send_message(chat_id, prompt_text, reply_markup=keyboard)
    else:
        # No existing info: default to replace mode
        business_info_update_pending[chat_id] = True
        business_info_mode[chat_id] = "replace"
        bot.answer_callback_query(call.id, "لطفاً اطلاعات بیزینس خود را به عنوان پیام ارسال کنید.")
        bot.send_message(chat_id, "📄 لطفاً اطلاعات بیزینس خود را ارسال کنید:")

# -------------------------------
# New Callback Handler: Replace Business Info
# -------------------------------
@bot.callback_query_handler(func=lambda call: call.data == "replace_business_info")
def handle_replace_business_info(call):
    chat_id = str(call.message.chat.id)
    business_info_update_pending[chat_id] = True
    business_info_mode[chat_id] = "replace"
    bot.answer_callback_query(call.id, "لطفاً اطلاعات جدید بیزینس را ارسال کنید تا جایگزین اطلاعات فعلی شود.")
    bot.send_message(chat_id, "📄 اطلاعات جدید بیزینس خود را ارسال کنید:")

# -------------------------------
# New Callback Handler: Append Business Info
# -------------------------------
@bot.callback_query_handler(func=lambda call: call.data == "append_business_info")
def handle_append_business_info(call):
    chat_id = str(call.message.chat.id)
    business_info_update_pending[chat_id] = True
    business_info_mode[chat_id] = "append"
    current_info = get_user_business_info(chat_id)
    bot.answer_callback_query(call.id, "لطفاً اطلاعات جدید بیزینس را ارسال کنید تا به اطلاعات فعلی اضافه شود.")
    bot.send_message(chat_id, f"📄 اطلاعات فعلی بیزینس:\n{current_info}\n\nاطلاعات جدید را ارسال کنید:")

# Updated AI Tone Button Handler with inline keyboard for tone selection.
@bot.callback_query_handler(func=lambda call: call.data == "ai_tone")
def handle_ai_tone(call):
    chat_id = str(call.message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_dostane = telebot.types.InlineKeyboardButton("دوستانه", callback_data="ai_tone_dostane")
    btn_rasmi = telebot.types.InlineKeyboardButton("رسمی", callback_data="ai_tone_rasmi")
    btn_professional = telebot.types.InlineKeyboardButton("حرفه ای و پروفشنال", callback_data="ai_tone_pro")
    keyboard.add(btn_dostane, btn_rasmi, btn_professional)
    bot.answer_callback_query(call.id)
    bot.send_message(chat_id, "لطفاً یکی از لحن‌های زیر را انتخاب کنید:", reply_markup=keyboard)

# New Callback Handler for selected AI Tone options.
@bot.callback_query_handler(func=lambda call: call.data in ["ai_tone_dostane", "ai_tone_rasmi", "ai_tone_pro"])
def select_ai_tone(call):
    chat_id = str(call.message.chat.id)
    tone_mapping = {
        "ai_tone_dostane": ("دوستانه", "genearte a efficient prompt to give in insiturction"),
        "ai_tone_rasmi": ("رسمی", "genearte a efficient prompt to give in insiturction"),
        "ai_tone_pro": ("حرفه ای و پروفشنال", "genearte a efficient prompt to give in insiturction"),
    }
    selected = tone_mapping.get(call.data)
    if selected:
        ai_tone_map[chat_id] = f"{selected[0]}: {selected[1]}"
        bot.answer_callback_query(call.id, f"لحن انتخاب شده: {selected[0]}")
        bot.send_message(chat_id, f"✅ لحن هوش مصنوعی به '{selected[0]}' تغییر یافت.")

# -------------------------------
# Telegram Content Handlers for Photo and Text
# -------------------------------
LAST_IMAGE_ANALYSIS = {}  # Stores analysis from the last received image per chat

# 1. Updated: Analyze image function now instructs LLM to respond in Persian.
def analyze_image(image, user_text=None):
    """
    Analyze the given image with optional user text input.
    """
    prompt = ""
    if user_text:
        prompt += f"متن ورودی کاربر: {user_text}\n\n"
    prompt += image_analyzer_prompt
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            image=image
        )
        # Handle different response formats
        if hasattr(response.choices[0].message, 'content'):
            return response.choices[0].message.content
        elif isinstance(response.choices[0].message, dict):
            return response.choices[0].message.get('content', '')
        else:
            return str(response.choices[0].message)
    except Exception as e:
        logging.error(f"Error in image analysis: {e}")
        return "خطا در تحلیل تصویر. لطفاً دوباره تلاش کنید."

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    chat_id = str(message.chat.id)
    sender_first_name = message.from_user.first_name or message.from_user.username
    user_text = message.caption if message.caption else ""
    logging.info("Received a photo message in chat '%s' from user '%s' with text: %s", chat_id, sender_first_name, user_text)
    
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        logging.info("Downloading image from URL: %s", file_url)
        
        bot.send_chat_action(chat_id, 'typing')
        placeholder_message = bot.reply_to(
            message,
            escape_markdown_v2("🤔 در حال تحلیل تصویر..."),
            parse_mode="MarkdownV2"
        )
        
        response = requests.get(file_url)
        if response.status_code == 200:
            # Analyze image with user text
            image_analysis = analyze_image(response.content, user_text)
            LAST_IMAGE_ANALYSIS[chat_id] = image_analysis
            
            # Use the same refinement process as text messages
            refined_response = refine_ai_response(image_analysis)
            save_message_to_history(chat_id, "user", f"{sender_first_name} sent an image" + (f" with text: {user_text}" if user_text else ""))
            save_message_to_history(chat_id, "assistant", refined_response)
            
            bot.edit_message_text(
                refined_response,
                chat_id=chat_id,
                message_id=placeholder_message.message_id,
                parse_mode="MarkdownV2"
            )
        else:
            bot.edit_message_text(
                escape_markdown_v2("❌ بارگیری تصویر موفقیت‌آمیز نبود."),
                chat_id=chat_id,
                message_id=placeholder_message.message_id,
                parse_mode="MarkdownV2"
            )
    except Exception as ex:
        error_message = escape_markdown_v2("❌ *خطا در پردازش تصویر. لطفاً دوباره تلاش کنید.*")
        bot.edit_message_text(
            error_message,
            chat_id=chat_id,
            message_id=placeholder_message.message_id,
            parse_mode="MarkdownV2"
        )
        logging.error("Exception in handle_photo for chat '%s': %s", chat_id, ex)

# 2. New helper to process and clean business info text.
def process_business_info(info_text, chat_id):
    # ساده‌سازی متن ورودی؛ حذف فاصله‌های اضافی.
    return info_text.strip()

def summarize_business_info(raw_text: str) -> str:
    """
    Summarize the provided business info in Persian, focusing on:
      - Main business domain & goals
      - Team members, roles, skills, responsibilities
      - Key features & analysis
    Limit output to ~200 words.
    """
    try:
        summary_prompt = (
            "لطفاً اطلاعات کسب‌وکار زیر را خلاصه و سازمان‌دهی کن. "
            "اطلاعات اصلی در مورد حوزه کاری، اهداف، تیم، نقش‌ها، توانایی‌ها و جوانب کلیدی را استخراج کن و آن را "
            "در حدود ۲۰۰ واژه در زبان فارسی توضیح بده:\n\n"
            f"{raw_text}"
        )
        response = llm_business([HumanMessage(content=summary_prompt)])
        return response.content.strip()
    except Exception as e:
        logging.error("Error summarizing business info: %s", e)
        return raw_text

# 3. Refined Business Info Update in text handler.
@bot.message_handler(func=lambda message: message.text is not None and not message.text.startswith("/"))
def handle_message(message):
    chat_type = message.chat.type
    chat_id = str(message.chat.id)
    # NEW: Check if the user is updating AI tone
    if ai_tone_update_pending.get(chat_id):
        new_tone = message.text.strip()
        ai_tone_map[chat_id] = new_tone
        ai_tone_update_pending.pop(chat_id, None)
        bot.send_message(chat_id, f"✅ لحن هوش مصنوعی به '{new_tone}' تغییر یافت.")
        return
    if business_info_update_pending.get(chat_id):
        new_info = process_business_info(message.text, chat_id)
        mode = business_info_mode.get(chat_id, "replace")
        if mode == "append":
            current_info = get_user_business_info(chat_id)
            combined_text = current_info + "\n" + new_info
            optimized = summarize_business_info(combined_text)
        else:
            optimized = summarize_business_info(new_info)
        
        # Save to permanent storage
        save_user_business_info(chat_id, optimized)
        business_info_update_pending.pop(chat_id, None)
        business_info_mode.pop(chat_id, None)
        bot.send_message(chat_id, "✅ اطلاعات بیزینس به‌روزرسانی شد.")
        bot.send_message(chat_id, f"🔍 اطلاعات بیزینس نهایی:\n{optimized}")
        return
    user_message_text = message.text
    sender_first_name = message.from_user.first_name or message.from_user.username
    sender_first_name = message.from_user.first_name or message.from_user.username
    logging.info("Received text message in chat '%s' from user '%s'.", chat_id, sender_first_name)
    
    save_message_to_history(chat_id, "user", f"{sender_first_name}: {user_message_text}")
    
    if chat_type in ['group', 'supergroup']:
        bot_username = bot.get_me().username
        is_mentioned = (
            (message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id) or
            (user_message_text and (bot_username in user_message_text)) or
            (message.entities and any(
                entity.type == 'mention' and user_message_text[entity.offset:entity.offset + entity.length].lower() == f"@{bot_username.lower()}"
                for entity in message.entities
            )) or ("بلو" in user_message_text)
        )
        if not is_mentioned:
            logging.info("Message in group '%s' stored but not triggering a response (bot not mentioned).", chat_id)
            return

    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.reply_to(
        message,
        escape_markdown_v2("🤔 در حال فکر کردن..."),
        parse_mode="MarkdownV2"
    )
    
    try:
        ai_tone = ai_tone_map.get(chat_id, "دوستانه")
        business_info = get_user_business_info(chat_id)  # Get from MongoDB instead of map
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
        
        prompt_input = f"{sender_first_name}: {user_message_text}"
        if chat_id in LAST_IMAGE_ANALYSIS:
            prompt_input += f"\n\n[remember you analyze this image and extracted this info , first analyze user request and then use related part from this Image Analysis , dont write un related to user query information in response  ]: {LAST_IMAGE_ANALYSIS[chat_id]}"
            LAST_IMAGE_ANALYSIS.pop(chat_id, None)
        full_prompt = prompt.format(input=prompt_input, ai_tone=ai_tone, business_info=business_info, history=history.messages)
        logging.info("Full prompt sent to AI:\n%s", full_prompt)
        
        ai_response = chain_with_history.invoke(
            {"input": prompt_input, "ai_tone": ai_tone, "business_info": business_info},
            config={"configurable": {"session_id": chat_id}}
        )
        logging.info("LangChain chain returned response for chat '%s'.", chat_id)
        refined_response = refine_ai_response(ai_response.content)
        save_message_to_history(chat_id, "assistant", refined_response)
        logging.info("Chat session '%s' now has %d messages in history.",
                     get_history_for_chat(chat_id).session_id,
                     len(get_history_for_chat(chat_id).messages))
        
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2(
            "❌ متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید."
        )
        bot.edit_message_text(
            error_message,
            chat_id=chat_id,
            message_id=placeholder_message.message_id,
            parse_mode="MarkdownV2"
        )
        logging.error("Error invoking LangChain chain for text message in chat '%s': %s", chat_id, e)

# 4. Refined Business Info Update in document handler.
@bot.message_handler(content_types=["document"])
def handle_document(message):
    chat_id = str(message.chat.id)
    if business_info_update_pending.get(chat_id):
        file_info = bot.get_file(message.document.file_id)
        file_bytes = bot.download_file(file_info.file_path)
        new_info = process_business_info(file_bytes.decode('utf-8'), chat_id)
        mode = business_info_mode.get(chat_id, "replace")
        if mode == "append":
            current_info = get_user_business_info(chat_id)
            optimized = current_info + "\n" + new_info
        else:
            optimized = new_info
        
        # Save to permanent storage
        save_user_business_info(chat_id, optimized)
        business_info_update_pending.pop(chat_id, None)
        business_info_mode.pop(chat_id, None)
        bot.send_message(chat_id, "✅ اطلاعات بیزینس به‌روزرسانی شد.")
        bot.send_message(chat_id, f"🔍 اطلاعات بیزینس نهایی:\n{optimized}")
    else:
        bot.reply_to(message, "دستور به‌روزرسانی اطلاعات بیزینس فعال نیست.")

# ============================================
# Main Function to Run the Bot
# ============================================
def main():
    logging.info("Setting up bot commands...")
    setup_bot_commands()
    logging.info("Telegram Bot is now running. Group conversations are logged and used as context.")
    bot.polling(none_stop=True)

if __name__ == "__main__":
    main()
