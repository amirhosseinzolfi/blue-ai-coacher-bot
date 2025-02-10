import os
import datetime
import logging
import threading
import atexit
import asyncio

# For Windows: use WindowsSelectorEventLoopPolicy to avoid certain warnings
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Monkey-patch Telebot API helper to disable SSL verification for testing only.
import telebot.apihelper as apihelper
import requests
apihelper.SESSION = requests.Session()
apihelper.SESSION.verify = False  # WARNING: Disable SSL verification ONLY for testing

from telebot.types import BotCommandScopeAllGroupChats, BotCommandScopeDefault, BotCommand
import telebot
from pymongo import MongoClient

# --- Set up API keys and Global Settings ---
TELEGRAM_BOT_TOKEN = "7796762427:AAGDTTAt6qn0-bTpnkejqsy8afQJLZhWkuk"  # Replace with your actual token
GOOGLE_API_KEY = "AIzaSyBAHu5yR3ooMkyVyBmdFxw-8lWyaExLjjE"           # Replace with your actual API key
OPENAI_API_KEY = "123"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global dictionaries for per-chat settings
chat_session_map = {}  # Maps Telegram chat id to current LangChain session id
business_info_map = {}  # Stores business info per chat_id
ai_tone_map = {}        # Stores AI tone per chat_id (default: "دوستانه")

# --- MongoDB History Configuration ---
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "chat_db"
COLLECTION_NAME = "chat_histories"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Start the G4F Interference API Server Automatically ---
try:
    from g4f.api import run_api
except ImportError:
    logging.error("g4f.api module not found. Please install the 'g4f' package.")
    run_api = None

if run_api is not None:
    def start_interference_api():
        logging.info("Starting G4F Interference API server on http://localhost:15203/v1 ...")
        run_api(bind="0.0.0.0:15203")
    api_thread = threading.Thread(target=start_interference_api, daemon=True)
    api_thread.start()

# --- Helper functions for MongoDB ---
def get_mongo_collection():
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    return db[COLLECTION_NAME]
 
def get_history_for_chat(telegram_chat_id: str):
    """
    Returns a MongoDBChatMessageHistory instance for the given Telegram chat.
    If no session exists, a new session id is created.
    """
    if telegram_chat_id not in chat_session_map:
        new_session_id = telegram_chat_id + "_" + str(int(datetime.datetime.now().timestamp()))
        chat_session_map[telegram_chat_id] = new_session_id
        logging.info(f"New session created: {new_session_id} for Telegram chat id {telegram_chat_id}")
    session_id = chat_session_map[telegram_chat_id]
    logging.info(f"Using session id {session_id} for Telegram chat id {telegram_chat_id}")
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    history_obj = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_CONNECTION_STRING,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
    )
    try:
        messages = history_obj.messages
        logging.info(f"Loaded {len(messages)} messages from history for session {session_id}")
    except Exception as e:
        logging.warning(f"Could not retrieve history messages for session {session_id}: {e}")
    return history_obj

def get_history_count(session_id: str):
    """Returns the number of messages stored for a given session id."""
    collection = get_mongo_collection()
    count = collection.count_documents({"session_id": {"$regex": f"^{session_id}"}})
    return count

# --- Telegram Bot Initialization ---
logging.info("Initializing Telegram bot...")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
logging.info("Telegram bot successfully initialized.")

# --- LangChain LLM Initialization ---
logging.info("Initializing LangChain LLM with OpenAI configuration...")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",  # Ensure your local API endpoint is running
    model_name="gpt-4o"
)
logging.info("LangChain LLM initialized.")

# --- New LLM for Business Info Summarization ---
logging.info("Initializing LangChain LLM with Gemini configuration for business info summarization...")
llm_business = ChatOpenAI(
    base_url="http://localhost:15203/v1",  # Ensure your local API endpoint is running
    model_name="gemini-1.5-flash" 
)
logging.info("LangChain Gemini LLM for business info summarization initialized.")

# Register an atexit callback to close the underlying aiohttp sessions
atexit.register(lambda: llm.client.close() if hasattr(llm, "client") and callable(getattr(llm.client, "close", None)) else None)
atexit.register(lambda: llm_business.client.close() if hasattr(llm_business, "client") and callable(getattr(llm_business.client, "close", None)) else None)

# --- Prompt Template for LLM ---
logging.info("Creating prompt template for LLM...")
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
(
    "system",
    "You are Blue, a professional AI business coach using the GPT-4o model. Your task is to provide precise, personalized advice for business and team growth.\n\n\
**Response Requirements:**\n\
- All answers must be in Markdown (use headings, subheadings, lists, tables, etc.).\n\
- Keep responses short, summarized, yet complete.\n\n\
**Guidelines:**\n\
- Carefully follow the user's requests; do only what the user asks, not just the system instructions.\n\
- Use the user's name in every response to foster a friendly tone.\n\
- Consider all previous instructions for coherent answers.\n\
- Maintain your tone as `{ai_tone}`.\n\
- The chat session is a group with several participants; analyze and follow all users' messages carefully and stay at the center of the conversation.\n\n\
Business and employee info: {business_info}\n\n\
**Important:** Answer all user messages in Persian."
),

    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
logging.info("Prompt template created.")

# --- Create a Chain with Message History using MongoDB ---
logging.info("Creating chain with message history (prompt → LLM)...")
from langchain_core.runnables.history import RunnableWithMessageHistory
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda telegram_chat_id: get_history_for_chat(telegram_chat_id),
    input_messages_key="input",
    history_messages_key="history",
)
logging.info("Chain with message history created.")

# --- Helper function for admin check ---
def is_admin(chat_id, user_id):
    try:
        member = bot.get_chat_member(chat_id, user_id)
        is_admin_status = member.status in ['creator', 'administrator']
        logging.debug(f"Admin check for user {user_id} in chat {chat_id}: {is_admin_status}")
        return is_admin_status
    except Exception as e:
        logging.error(f"Error checking admin status for user {user_id} in chat {chat_id}: {e}")
        return False

def setup_bot_commands():
    commands = [
        BotCommand("start", "شروع ربات و نمایش اطلاعات چت"),
        BotCommand("options", "انتخاب گزینه‌ها"),
        BotCommand("new_chat", "ایجاد جلسه چت جدید"),
        BotCommand("history", "نمایش تاریخچه جلسات"),
        BotCommand("help", "نمایش پیام راهنما"),
        BotCommand("settings", "تنظیمات ربات"),
        BotCommand("about", "اطلاعات ربات")
    ]
    try:
        logging.info("Setting up bot commands for private and group chats...")
        try:
            bot.delete_my_commands(scope=BotCommandScopeDefault())
            bot.delete_my_commands(scope=BotCommandScopeAllGroupChats())
        except Exception as e:
            logging.error(f"Error deleting existing bot commands: {e}")
        bot.set_my_commands(commands, scope=BotCommandScopeDefault())
        bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())
        logging.info("Bot commands have been set up successfully.")
    except Exception as e:
        logging.error(f"Exception during bot command setup: {e}")

# --- Command Handlers ---
@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = str(message.chat.id)
    welcome_message = (
        "سلام! من بلو هستم، همراه هوشمند کسب‌وکار تو! 🚀✨\n\n"
        "من با استفاده از هوش مصنوعی قدرتمند GPT-4o و تحلیل دقیق کسب‌وکار و گفت‌وگوهای تو، به عنوان یک مربی حرفه‌ای کنارت هستم.\n\n\n\n"

        "برای دریافت مشاوره شخصی‌سازی‌شده، ابتدا اطلاعات بیزینس و تیم خود را با دستور /settings ثبت کن.⚙️\n\n\n"
        "**چیکار میتونم بکنم من ؟ **\n\n"
        "🟢  برنامه‌ریزی وظایف تیم بر اساس مهارت‌های فردی\n"
        "🟢  ارائه گزارش‌های روزانه و تحلیل عملکرد تیم (/options)\n"
        "🟢  راهنمایی استراتژیک برای اجرای پروژه‌ها\n"
        "🟢  پاسخ‌گویی هوشمند بر اساس داده‌های اختصاصی کسب‌وکار تو\n"
        "🟢 تطبیق لحن پاسخ‌ها با تنظیمات انتخابی (/settings)\n\n"
  
        " **برای تعامل راحت‌ تر:**\n"
        "🔹 منو صدا بزن ('بلو')\n"
        "🔹 منو تگ کن (@Blue)\n"
        "🔹 از گزینه‌های بات استفاده کن\n\n"
        " فقط کافیه صدام کنی  😉\n"
    )
    bot.reply_to(message, welcome_message, parse_mode="Markdown")
    logging.info(f"/start command processed for Chat ID: {chat_id}")

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = (
        "🤖 *دستورات ربات:*\n\n"
        " - `/start` - شروع ربات و نمایش اطلاعات چت\n"
        " - `/options` - انتخاب گزینه‌ها\n"
        " - `/settings` - تنظیمات ربات\n"
        " - `/new_chat` - ایجاد جلسه چت جدید\n"
        " - `/history` - نمایش تاریخچه جلسات\n"
        " - `/help` - نمایش پیام راهنما\n"
        " - `/about` - اطلاعات ربات\n\n"
        "در گروه‌ها، من تنها زمانی پاسخ می‌دهم که منشن شوم یا کلمه *بلو* در پیام وجود داشته باشد."
    )
    bot.reply_to(message, help_text, parse_mode="Markdown")
    logging.info("/help command processed.")

@bot.message_handler(commands=['about'])
def about_bot(message):
    about_text = (
        "🤖 *درباره ربات:*\n\n"
        "من **بلو** هستم، مربی کسب‌وکار هوشمند با پشتیبانی از فناوری LangChain و مدل‌های OpenAI.\n"
        "برای اطلاعات بیشتر از دستور `/help` استفاده کنید."
    )
    bot.reply_to(message, about_text, parse_mode="Markdown")
    logging.info(f"/about command processed for Chat ID: {message.chat.id}")

@bot.message_handler(commands=['settings'])
def bot_settings(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_business_info = telebot.types.InlineKeyboardButton("بارگذاری اطلاعات بیزینس", callback_data="load_business_info")
    btn_ai_tone = telebot.types.InlineKeyboardButton("انتخاب لحن هوش مصنوعی", callback_data="ai_tone")
    keyboard.add(btn_business_info, btn_ai_tone)
    settings_text = (
        "⚙️ *تنظیمات ربات:*\n\n"
        "لطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
    )
    bot.reply_to(message, settings_text, reply_markup=keyboard, parse_mode="Markdown")
    logging.info(f"/settings command processed for Chat ID: {message.chat.id}")

@bot.message_handler(commands=['new_chat'])
def new_chat(message):
    chat_id = str(message.chat.id)
    new_session_id = chat_id + "_" + str(int(datetime.datetime.now().timestamp()))
    chat_session_map[chat_id] = new_session_id
    bot.reply_to(message, f"💡 *جلسه چت جدید آغاز شد!*\n(Session ID: `{new_session_id}`)", parse_mode="Markdown")
    logging.info(f"/new_chat command processed for chat {chat_id}. New session ID: {new_session_id}")

@bot.message_handler(commands=['history'])
def show_history_sessions(message):
    chat_id = str(message.chat.id)
    collection = get_mongo_collection()
    sessions = collection.distinct("session_id", {"session_id": {"$regex": f"^{chat_id}_"}})
    logging.info(f"Distinct session IDs from MongoDB: {sessions}")
    if not sessions:
        bot.reply_to(message, "هیچ جلسه چتی در تاریخچه یافت نشد.", parse_mode="Markdown")
        logging.info(f"/history command: No sessions found for chat {chat_id}")
        return
    keyboard = telebot.types.InlineKeyboardMarkup()

    for session in sessions:
        parts = session.split("_")
        if len(parts) == 2:
            try:
                timestamp = int(parts[1])
                date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                date_str = "نامشخص"
        else:
            date_str = "جلسه پیش‌فرض"
        button_text = f"جلسه: {date_str}"
        keyboard.add(telebot.types.InlineKeyboardButton(button_text, callback_data=f"load_session:{session}"))
    bot.reply_to(message, "📜 *تاریخچه جلسات:*", reply_markup=keyboard, parse_mode="Markdown")
    logging.info(f"/history command: Found {len(sessions)} sessions for chat {chat_id}")

@bot.callback_query_handler(func=lambda call: call.data.startswith("load_session:"))
def load_session_callback(call):
    session_id = call.data.split("load_session:")[1]
    chat_id = str(call.message.chat.id)
    chat_session_map[chat_id] = session_id
    history_obj = get_history_for_chat(chat_id)
    count = len(history_obj.messages)
    bot.answer_callback_query(call.id, text=f"جلسه `{session_id}` با {count} پیام بارگذاری شد.")
    bot.send_message(chat_id, f"*جلسه انتخاب شد:*\n`{session_id}`\n*تعداد پیام‌های ذخیره شده:* {count}", parse_mode="Markdown")
    logging.info(f"Session {session_id} loaded for chat {chat_id} with {count} messages.")

@bot.message_handler(commands=['options'])
def options_command(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_daily_report = telebot.types.InlineKeyboardButton("آنالیز روزانه", callback_data="option_report_daily")
    btn_analyze_members = telebot.types.InlineKeyboardButton("آنالیز امروز اعضا", callback_data="option_analyze_today")
    btn_tasks_today = telebot.types.InlineKeyboardButton("تسک‌های امروز", callback_data="option_tasks_today")
    keyboard.add(btn_daily_report, btn_analyze_members, btn_tasks_today)
    bot.reply_to(message, "🤔 *چه کمکی از دست من برمی‌آید؟*", reply_markup=keyboard, parse_mode="Markdown")
    logging.info(f"/options command processed for Chat ID: {message.chat.id}")

@bot.message_handler(content_types=['new_chat_members'])
def on_new_chat_member(message):
    if message.chat.type in ['group', 'supergroup']:
        for new_member in message.new_chat_members:
            if new_member.id == bot.get_me().id:
                chat_id = str(message.chat.id)
                welcome_text = (
                    f"سلام دوستان!\n\n"
                    f"من **بلو** هستم، جهت استفاده از من از `@{bot.get_me().username}` استفاده کنید."
                )
                bot.reply_to(message, welcome_text, parse_mode="Markdown")
                logging.info(f"Bot added to group: {chat_id}")
                break
    elif message.chat.type == 'private':
        bot.reply_to(message, "سلام! من **بلو** هستم، مربی حرفه‌ای کسب‌وکار. خوش آمدید!", parse_mode="Markdown")
        logging.info(f"Bot started in private chat: {message.chat.id}")

# --- Callback Query Handlers for Inline Settings ---
@bot.callback_query_handler(func=lambda call: call.data == "load_business_info")
def handle_load_business_info(call):
    bot.answer_callback_query(call.id)
    logging.info(f"Loading business info for chat {call.message.chat.id}")
    msg = bot.send_message(call.message.chat.id, "📝 *لطفاً اطلاعات کسب‌وکار و کارکنان خود را وارد کنید:*", parse_mode="Markdown")
    bot.register_next_step_handler(msg, process_business_info)

def process_business_info(message):
    chat_id = str(message.chat.id)
    raw_info = message.text
    # Create summarization prompt for the business info
    prompt_text = (
        "لطفاً اطلاعات کسب‌وکار زیر را به صورت ساختارمند و بهینه خلاصه کن، بدون اینکه اطلاعات مهم حذف شود:\n\n"
        f"{raw_info}"
    )
    try:
        # Use the Gemini LLM to summarize and structure the business info
        # Pass the prompt as a plain string instead of a dict.
        summarized_response = llm_business.invoke(prompt_text)
        summarized_info = summarized_response.content if hasattr(summarized_response, "content") else str(summarized_response)
        # Save the optimized business info (replacing the raw user input)
        business_info_map[chat_id] = summarized_info
        # Send the optimized information back to the user
        bot.send_message(chat_id, f"✅ *اطلاعات کسب‌وکار بهینه شد:*\n\n{summarized_info}", parse_mode="Markdown")
        logging.info(f"Business info loaded for chat {chat_id} and summarized: {summarized_info}")
    except Exception as e:
        bot.send_message(chat_id, "❌ *خطا در پردازش اطلاعات کسب‌وکار. لطفاً دوباره تلاش کنید.*", parse_mode="Markdown")
        logging.error(f"Error processing business info for chat {chat_id}: {e}")

@bot.callback_query_handler(func=lambda call: call.data == "ai_tone")
def handle_ai_tone(call):
    bot.answer_callback_query(call.id)
    logging.info(f"Setting AI tone for chat {call.message.chat.id}")
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_formal = telebot.types.InlineKeyboardButton("رسمی", callback_data="set_ai_tone_رسمی")
    btn_friendly = telebot.types.InlineKeyboardButton("دوستانه", callback_data="set_ai_tone_دوستانه")
    btn_professional = telebot.types.InlineKeyboardButton("حرفه‌ای", callback_data="set_ai_tone_حرفه‌ای")
    keyboard.add(btn_formal, btn_friendly, btn_professional)
    bot.send_message(call.message.chat.id, "🔧 *لطفاً لحن صحبت هوش مصنوعی را انتخاب کنید:*", reply_markup=keyboard, parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith("set_ai_tone_"))
def set_ai_tone(call):
    tone = call.data.split("set_ai_tone_")[1]
    chat_id = str(call.message.chat.id)
    ai_tone_map[chat_id] = tone
    bot.answer_callback_query(call.id, text=f"لحن به *{tone}* تغییر یافت.")
    bot.send_message(chat_id, f"✅ *لحن هوش مصنوعی به `{tone}` تنظیم شد.*", parse_mode="Markdown")
    logging.info(f"AI tone set for chat {chat_id}: {tone}")

# --- Callback Query Handlers for Options Commands ---
def process_option_prompt(chat_id, prompt_text):
    logging.info(f"Processing option prompt for chat {chat_id}: {prompt_text}")
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, "🤔 *در حال پردازش...*", parse_mode="Markdown")
    try:
        ai_response = chain_with_history.invoke(
            {
                "input": prompt_text,
                "ai_tone": ai_tone_map.get(chat_id, "دوستانه"),
                "business_info": business_info_map.get(chat_id, "")
            },
            config={"configurable": {"session_id": chat_id}}
        )
        logging.info(f"AI response for chat {chat_id}: {ai_response.content}")
        history_obj = get_history_for_chat(chat_id)
        count = len(history_obj.messages)
        logging.info(f"Chat session {history_obj.session_id} now has {count} messages saved in MongoDB.")
        bot.edit_message_text(ai_response.content, chat_id=chat_id, message_id=placeholder_message.message_id, parse_mode="Markdown")
    except Exception as e:
        error_message = (
            "❌ *متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.*\n\n"
            f"*Error:* `{str(e)}`"
        )
        bot.edit_message_text(error_message, chat_id=chat_id, message_id=placeholder_message.message_id, parse_mode="Markdown")
        logging.error(f"Error invoking chain for chat {chat_id}: {e}")

@bot.callback_query_handler(func=lambda call: call.data == "option_report_daily")
def handle_option_report_daily(call):
    bot.answer_callback_query(call.id)
    chat_id = str(call.message.chat.id)
    prompt_text = (
        "⚡ لطفاً کل تاریخچه این چت را به عنوان منبع بررسی در نظر بگیر و تنها پیام‌های مرتبط با فعالیت‌های کاری و تسک‌ها را تحلیل کن.\n\n"
        "## ۱. تحلیل عملکرد تیم:\n"
        "- **فعالیت‌های روزانه:** تحلیل روند فعالیت‌های تیم در طول روز\n"
        "- **آمار موفقیت/شکست:** ارائه آمار مرتبط با موفقیت‌ها و شکست‌ها\n\n"
        "## ۲. تسک‌های امروز:\n"
        "- تهیه لیستی ((چدول نباشه به صورت لیست ))از تسک‌های امروز به همراه توضیحات مربوط به هر تسک\n\n"
        "## ۳. ارزیابی عملکرد:\n"
        "- **امتیاز کلی تیم:** ارائه نمره‌ای از ۱ تا ۱۰ به همراه توضیحات مربوط به نقاط قوت و ضعف\n\n"
        "💡 لطفاً گزارش را به صورت ساختارمند، خوانا و جذاب (با استفاده از علائم و استیکرهای مناسب) به روش استاندارد Markdown v2 ارائه کن."
    )
    logging.info(f"Option 'گزارش روزانه تیم' selected for chat {chat_id}.")
    process_option_prompt(chat_id, prompt_text)


@bot.callback_query_handler(func=lambda call: call.data == "option_analyze_today")
def handle_option_analyze_today(call):
    bot.answer_callback_query(call.id)
    chat_id = str(call.message.chat.id)
    prompt_text = (
        "⚡ لطفاً تاریخچه این چت را مرور کن و پیام‌های مرتبط با فعالیت‌های کاری و تسک‌ها را تحلیل کن.\n\n"
        "### برای هر کاربر یک گزارش جداگانه تهیه کن که شامل موارد زیر باشد:\n"
        "- **امتیازدهی:** ارائه نمره‌ای از ۱ تا ۱۰\n"
        "- **بینش‌های عمیق:** توضیح نقاط قوت و ضعف به صورت مختصر\n"
        "- **لیست وظایف:** نمایش وظایف انجام‌شده و در حال انجام به شکل منظم (بدون استفاده از جدول)\n\n"
        "💡 لطفاً گزارش‌ها را به صورت سازمان‌یافته و با استفاده از ساختار Markdown v2 و علائم جذاب ارائه کن."
    )
    logging.info(f"Option 'آنالیز امروز اعضا' selected for chat {chat_id}.")
    process_option_prompt(chat_id, prompt_text)


@bot.callback_query_handler(func=lambda call: call.data == "option_tasks_today")
def handle_option_tasks_today(call):
    bot.answer_callback_query(call.id)
    chat_id = str(call.message.chat.id)
    prompt_text = (
        "🔎 بر اساس تاریخچه پیام‌های این چت، برای هر کاربر یک لیست از تسک‌ها تهیه کن.\n\n"
        "لطفاً برای هر تسک، اطلاعات زیر را به‌طور منظم (چدول نباشه به صورت لیست ) ارائه بده:\n"
        "- **تسک:** شرح کوتاه و واضح تسک\n"
        "- **مسئول:** نام فرد مسئول\n"
        "- **زمان مورد نیاز:** تخمین زمان لازم برای انجام تسک\n"
        "- **وضعیت:** مشخص کن که تسک انجام شده است یا خیر\n\n"
        "💡 نتایج را در قالب یک ساختار Markdown v2 منظم (با استفاده از سرخط‌ها و لیست‌های بولت) ارائه کن. از استفاده از جداول خودداری کن و از علائم و استیکرهای جذاب بهره ببر."
    )
    logging.info(f"Option 'تسک‌های امروز' selected for chat {chat_id}.")
    process_option_prompt(chat_id, prompt_text)

@bot.message_handler(func=lambda message: message.text is not None and not message.text.startswith("/"))
def handle_message(message):
    """
    پردازش پیام‌های متنی (غیر از دستورات).
    در گروه‌ها، ربات تنها زمانی پاسخ می‌دهد که منشن شود یا کلمه "بلو" در پیام وجود داشته باشد.
    """
    chat_type = message.chat.type
    chat_id = str(message.chat.id)
    user_message_text = message.text
    sender_first_name = message.from_user.first_name if message.from_user.first_name else message.from_user.username
    modified_user_message = f"*اسم کاربر:* `{sender_first_name}`\n{user_message_text}"
    logging.info(f"User prompt in chat {chat_id}: {modified_user_message}")

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
        logging.info(f"Bot not mentioned in group chat {chat_id}; no reply will be sent.")
        return

    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.reply_to(message, "🤔 *در حال پردازش...*", parse_mode="Markdown")
    try:
        ai_response = chain_with_history.invoke(
            {
                "input": modified_user_message,
                "ai_tone": ai_tone_map.get(chat_id, "دوستانه"),
                "business_info": business_info_map.get(chat_id, "")
            },
            config={"configurable": {"session_id": chat_id}}
        )
        logging.info(f"AI response for chat {chat_id}: {ai_response.content}")
        history_obj = get_history_for_chat(chat_id)
        count = len(history_obj.messages)
        logging.info(f"Chat session {history_obj.session_id} now has {count} messages saved in MongoDB.")
        bot.edit_message_text(ai_response.content, chat_id=chat_id, message_id=placeholder_message.message_id, parse_mode="Markdown")
    except Exception as e:
        error_message = (
            "❌ *متأسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.*\n\n"
            f"*Error:* `{str(e)}`"
        )
        bot.edit_message_text(error_message, chat_id=chat_id, message_id=placeholder_message.message_id, parse_mode="Markdown")
        logging.error(f"Error invoking chain for chat {chat_id}: {e}")

def main():
    setup_bot_commands()
    logging.info("🚀 Telegram Bot is running. Chat sessions are managed via LangChain’s MongoDB memory mechanism...")
    bot.polling(none_stop=True)

if __name__ == "__main__":
    main()



