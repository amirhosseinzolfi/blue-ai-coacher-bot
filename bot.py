import os
import datetime
import logging
import threading
from sqlalchemy import create_engine, text
import telebot
from telebot.types import BotCommandScopeAllGroupChats, BotCommandScopeDefault, BotCommand

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

# --- Set up API keys ---
TELEGRAM_BOT_TOKEN = "7796762427:AAGDTTAt6qn0-bTpnkejqsy8afQJLZhWkuk"  # Replace with your actual token
GOOGLE_API_KEY = "AIzaSyBAHu5yR3ooMkyVyBmdFxw-8lWyaExLjjE"           # Replace with your actual API key
OPENAI_API_KEY = "123"
# Global dictionaries for per-chat settings
chat_session_map = {}
business_info_map = {}  # Stores business info per chat_id
ai_tone_map = {}        # Stores AI tone per chat_id (default: "دوستانه")

# IMPORTANT: Set the base_url to the locally running API server.
OPENAI_BASE_URL = "http://localhost:15203/v1"
OPENAI_MODEL_NAME = "gpt-4o-mini"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Database Initialization for Chat History ---
logging.info("Initializing SQLite database for chat history...")
engine = create_engine("sqlite:///telegram_chat_history.db")
logging.info("SQLite engine created using database 'telegram_chat_history.db'.")

def initialize_history_table():
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_message_histories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        connection.commit()
    logging.info("Table 'chat_message_histories' ensured in the database.")

initialize_history_table()

def get_message_history(session_id: str):
    actual_session_id = chat_session_map.get(session_id, session_id)
    logging.info(f"Retrieving message history for session: {actual_session_id}.")
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    history_obj = SQLChatMessageHistory(session_id=actual_session_id, connection=engine)
    try:
        messages = history_obj.get_messages()  # Assuming get_messages() exists
        logging.info(f"Loaded {len(messages)} messages from history for session: {actual_session_id}.")
    except Exception as e:
        logging.warning(f"Could not retrieve history messages count for session {actual_session_id}: {e}")
    return history_obj

# --- Telegram Bot Initialization ---
logging.info("Initializing Telegram bot...")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
logging.info("Telegram bot successfully initialized.")

# --- LangChain LLM Initialization ---
logging.info("Initializing LangChain LLM with OpenAI configuration...")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url=OPENAI_BASE_URL,
    model_name=OPENAI_MODEL_NAME
)
logging.info(f"LangChain LLM initialized using model '{OPENAI_MODEL_NAME}' at '{OPENAI_BASE_URL}'.")

# --- Prompt Template for LLM ---
logging.info("Creating prompt template for LLM...")
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "شما بلو (blue) هستید، یک مربی کسب و کار هوش مصنوعی حرفه‌ای، متخصص و بسیار مجرب که از قابلیت‌های پیشرفته مدل هوش مصنوعی GPT-4o استفاده می‌کنید. وظیفه اصلی شما مربیگری کسب و کارها و تیم‌هایشان برای دستیابی به عملکرد و رشد مطلوب است.\n\n"
        "**مسئولیت‌های اصلی شما شامل:**\n\n"
        "- **تحلیل عملکرد تیم:** تحلیل فعالیت‌های روزانه تیم و ارائه امتیازهای عملکرد فردی به اعضای تیم، ارائه بینش در مورد مشارکت‌هایشان و زمینه‌های قابل بهبود.\n"
        "- **تحلیل وظایف:** بررسی دقیق وظایف محول شده به هر عضو تیم برای درک توزیع حجم کار و پیچیدگی وظایف.\n"
        "- **سازماندهی و تخصیص وظایف:** سازماندهی و تخصیص استراتژیک وظایف به اعضای تیم، با در نظر گرفتن شخصیت‌های فردی، نقش‌هایشان در کسب و کار، و مجموعه‌های مهارتی برای به حداکثر رساندن کارایی و رضایت شغلی.\n"
        "- **راهنمایی حرفه‌ای کسب و کار:** ارائه مشاوره تخصصی کسب و کار، با استفاده از اطلاعات، اسناد و داده‌های کارمندان ارائه شده کسب و کار برای اطمینان از مرتبط بودن و دقت. تمرکز بر تصمیم‌گیری‌های استراتژیک، بهینه‌سازی فرآیندها، و پرورش یک محیط کار سالم و سازنده.\n"
        "- **مربیگری شخصی‌سازی شده و مدیریت وظایف:** ارائه استراتژی‌های مربیگری فردی و مدیریت وظایف برای هر عضو تیم. توصیه‌های خود را با نقش‌های خاص، شخصیت‌ها و داده‌های عملکرد آنها تنظیم کنید تا توسعه حرفه‌ای و مشارکت آنها در اهداف کسب و کار تسهیل شود.\n"
        "- **استخراج داده از تصویر:** تحلیل و استخراج داده‌های مرتبط از تصاویر ارائه شده، با درک اینکه این داده‌ها ممکن است حاوی اطلاعات یا بینش‌های حیاتی کسب و کار باشند.\n\n"
        "**دستورالعمل‌های مهم:**\n\n"
        "- **شخصی‌سازی و صمیمیت:** در هر پاسخ، تلاش آگاهانه‌ای برای استفاده از نام کاربر برای ایجاد یک تعامل شخصی‌تر و دوستانه‌تر انجام دهید. این به ایجاد رابطه و اعتماد کمک می‌کند.\n"
        "لطفاً در پاسخ‌هات از ساختار و نحو مارک‌داون استفاده نکن. برای ساختاردهی به متن‌ها از متن ساده، ایموجی‌ها و علائم استفاده کن"
        "- **آگاهی از زمینه:** به تمام جنبه‌های مکالمه ما، از جمله تمام پیام‌ها و دستورالعمل‌های قبلی، توجه دقیق داشته باشید. زمینه را در طول تعاملات خود حفظ کنید تا پاسخ‌های منسجم و مرتبط ارائه دهید.\n"
        "- **پاسخ‌های مبتنی بر داده:** به طور فعال اطلاعات کسب و کار، اسناد کسب و کار و اطلاعات کارمندان ارائه شده را هنگام تدوین پاسخ‌ها و توصیه‌های خود در نظر بگیرید و از آنها استفاده کنید. این اطمینان می‌دهد که راهنمایی شما بسیار مرتبط و به طور خاص برای زمینه کسب و کار طراحی شده است.\n"
        "- **لحن و رفتار:** لحن و رفتار {ai_tone} را اتخاذ کنید. این لحن ثابت را در طول تمام تعاملات حفظ کنید.\n"
        "- **دسترسی به داده:** شما به اطلاعات کسب و کار، اسناد کسب و کار و اطلاعات کارمندان زیر دسترسی دارید که برای مربیگری شما بسیار حیاتی است: . از این اطلاعات  برای بهبود کیفیت و ارتباط مربیگری خود و همچنین شخصی سازی بیشتر استفاده کنید.\n\n"
        "**اطلاعات بیزینس و کارمندان** : {business_info}"
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
logging.info("Prompt template created.")

# --- Chain with Message History ---
logging.info("Creating chain with message history (prompt → LLM)...")
from langchain_core.runnables.history import RunnableWithMessageHistory
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)
logging.info("Chain with message history successfully created.")

# --- Helper Functions ---
def is_admin(chat_id, user_id):
    try:
        member = bot.get_chat_member(chat_id, user_id)
        is_admin_status = member.status in ['creator', 'administrator']
        logging.debug(f"Admin check for user {user_id} in chat {chat_id}: {is_admin_status}.")
        return is_admin_status
    except Exception as e:
        logging.error(f"Error checking admin status for user {user_id} in chat {chat_id}: {e}.")
        return False

def setup_bot_commands():
    commands = [
        BotCommand("start", "شروع ربات و نمایش اطلاعات چت"),
        BotCommand("options", "انتخاب گزینه‌ها"),

        #BotCommand("getchatid", "دریافت شناسه چت"),
        #BotCommand("daily_report", "نمایش گزارش روزانه چت"),
        BotCommand("new_chat", "ایجاد جلسه چت جدید"),
        #BotCommand("show_sessions", "نمایش تمامی جلسات ذخیره شده گروه"),
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
        logging.error(f"Exception during bot command setup: {e}.")

# --- Command Handlers ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = str(message.chat.id)
    chat_type = message.chat.type
    welcome_message = (
        "سلام! من بلو هستم، همراه هوشمند کسب‌وکارت 😉\n\n"
        "خب ، اول اطلاعات کسب‌وکارت و تیمت رو توی /settings وارد کن تا بتونم مفید تر و به صورت کاملا شخصی سازی شده کمکت کنم.\n\n"
        "من با استفاده از هوش مصنوعی قدرتمند gpt-4o و آنالیز اطلاعات کامل بیزینس و آنالیز تمامی چت های شما قراره به صورت یه مربی حرفه ای کاملا شخصی سازی شده و هوشمند کسب و کار، کنار شما باشم !\n\n"
        "من چه کارایی میتونم انجام بدم؟\n\n"
        "ببین من تقریبا هرکاری بخوای میتونم انجام بدم ، مثل :"
        "* میتونم وظایف تیم رو بر اساس توانایی هاشون برنامه ریزی کنم\n"
        "* در انتهای هر روز یک آنالیز کامل از تسک ها، فعالیت های اعضای تیم به همراه نمره شخصی برای هر فرد مینویسم /options\n"
        "- میتونم گزارش فعالیت روزانه تیم رو نشونتون دهم /options\n"
        "- میتونم با آنالیز پروژه هاتون تو مسیر اجرایی کردن پروژه ها راهنمایی و کمکتون کنم\n"
        "- میتونم طبق اطلاعات و دیتا شخصی سازی شده بیزینس شما به سوالاتون جواب بدم\n"
        "- میتونم با هر لحنی که شما در تنظیمات انتخاب کنید صحبنت کنم  /settings\n"
        "- میتونم اطالاعات بیزینس شما و اعضای تیم رو آنالیز کنم و پاسخ های کاملا شخصی سازی شده بدم  /settings\n"
        "- \n"
        "و هر کار دیگه ای که فکرشو کنی فقط کافیه صدام کنی و ازم بپرسی! 😉\n\n\n"
        " فقط کافیه صدام کنی 'بلو' و ازم بپرسی! 😉\n\n\n"
        "(برای ارتباط با من، 'بلو' رو صدا کن، تگ کن یا از گزینه های بات استفاده کن.)"
    )
    bot.reply_to(message, welcome_message)
    logging.info(f"/start command processed for Chat ID: {chat_id}, Type: {chat_type}.")

@bot.message_handler(commands=['getchatid'])
def get_chat_id(message):
    chat_id = str(message.chat.id)
    chat_type = message.chat.type
    response = f"Chat ID: {chat_id}\nChat Type: {chat_type}"
    bot.reply_to(message, response)
    logging.info(f"/getchatid command processed for Chat ID: {chat_id}, Type: {chat_type}.")

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = (
        "🤖 دستورات ربات:\n"
        "/start - شروع ربات و نمایش اطلاعات چت\n"
        "/options - انتخاب گزینه‌ها\n"
        "/settings - تنظیمات ربات\n"
        "/new_chat - ایجاد جلسه چت جدید\n"
        "/help - نمایش پیام راهنما\n"
        #"/getchatid - دریافت شناسه و نوع چت فعلی\n"
        #"/daily_report - نمایش گزارش روزانه چت\n"

        #"/show_sessions - نمایش تمامی جلسات ذخیره شده گروه\n"
        "/about - اطلاعات ربات\n"


        "\nبرای ارسال پیام کافیست یک پیام متنی ارسال کنید. در گروه‌ها، من فقط زمانی پاسخ می‌دهم که منشن شوم یا از کلمه 'بلو' استفاده کنید."
    )
    bot.reply_to(message, help_text)
    logging.info("/help command processed.")

@bot.message_handler(commands=['about'])
def about_bot(message):
    about_text = (
        "🤖 درباره ربات:\n"
        "من بلو هستم، یک مربی کسب‌وکار حرفه‌ای که با استفاده از فناوری LangChain و مدل‌های OpenAI به شما کمک می‌کنم.\n"
        "برای اطلاعات بیشتر، می‌توانید از دستور /help استفاده کنید."
    )
    bot.reply_to(message, about_text)
    logging.info(f"/about command processed for Chat ID: {message.chat.id}.")

@bot.message_handler(commands=['settings'])
def bot_settings(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_business_info = telebot.types.InlineKeyboardButton("بار گزاری اطلاعات بیزینس", callback_data="load_business_info")
    btn_ai_tone = telebot.types.InlineKeyboardButton("انتخاب لحن صحبت هوش مصنوعی", callback_data="ai_tone")
    keyboard.add(btn_business_info, btn_ai_tone)
    settings_text = "⚙️ تنظیمات ربات:\nلطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
    bot.reply_to(message, settings_text, reply_markup=keyboard)
    logging.info(f"/settings command processed for Chat ID: {message.chat.id}.")

@bot.message_handler(commands=['options'])
def options_command(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_daily_report = telebot.types.InlineKeyboardButton("آنالیز روزانه", callback_data="option_report_daily")
    btn_analyze_members = telebot.types.InlineKeyboardButton("آنالیز امروز اعضا", callback_data="option_analyze_today")
    btn_tasks_today = telebot.types.InlineKeyboardButton("تسک های امروز", callback_data="option_tasks_today")
    keyboard.add(btn_daily_report)
    keyboard.add(btn_analyze_members)
    keyboard.add(btn_tasks_today)
    bot.reply_to(message, "میخوای چیکار کنم برات ؟", reply_markup=keyboard)
    logging.info(f"/options command processed for Chat ID: {message.chat.id}.")

@bot.message_handler(content_types=['new_chat_members'])
def on_new_chat_member(message):
    if message.chat.type in ['group', 'supergroup']:
        for new_member in message.new_chat_members:
            if new_member.id == bot.get_me().id:
                chat_id = str(message.chat.id)
                welcome_text = (
                    "سلام دوستان! من بلو هستم، خوشحالم که به گروه اضافه شدم! "
                    "از این به بعد، می‌توانید با من در مورد تسکاتون ، ایده پردازی کسب‌وکار و پروژه‌ها یا هر سوال دیگه که تو ذهنتونه صحبت کنید. "
                    f"برای منشن کردن از @{bot.get_me().username} استفاده کنید."
                )
                bot.reply_to(message, welcome_text)
                logging.info(f"Bot added to group: {chat_id}.")
                break
    elif message.chat.type == 'private':
        bot.reply_to(
            message,
            "سلام! من بلو هستم، یک مربی حرفه‌ای کسب‌وکار. خوشحالم که به چت من پیوستید! "
            "بیایید با هم درباره کسب‌وکار و پروژه‌ها صحبت کنیم."
        )
        logging.info(f"Bot started in private chat: {message.chat.id}.")

# --- Updated Command Handlers for Daily Report and New Chat Sessions ---

@bot.message_handler(commands=['daily_report'])
def daily_report(message):
    chat_id = str(message.chat.id)
    preset_input = "یه گزارش کامل از پیام های این چت تا لان بکو"
    logging.info(f"/daily_report triggered in chat {chat_id} with preset input: {preset_input}")

    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.reply_to(message, "🤔 در حال فکر کردن...")
    logging.info(f"Session ID for /daily_report: {chat_id} -> {chat_session_map.get(chat_id, chat_id)}")

    try:
        ai_response = chain_with_history.invoke(
            {
                "input": preset_input,
                "ai_tone": ai_tone_map.get(chat_id, "دوستانه"),
                "business_info": business_info_map.get(chat_id, "")
            },
            config={"configurable": {"session_id": chat_id}}
        )
        logging.info(f"AI response for /daily_report in chat {chat_id}: {ai_response.content}")
        bot.edit_message_text(ai_response.content, chat_id=chat_id, message_id=placeholder_message.message_id)
    except Exception as e:
        error_message = (
            f"متاسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.\n\nError: {str(e)}"
        )
        bot.edit_message_text(error_message, chat_id=chat_id, message_id=placeholder_message.message_id)
        logging.error(f"Error in /daily_report for chat {chat_id}: {e}")

@bot.message_handler(commands=['new_chat'])
def new_chat(message):
    chat_id = str(message.chat.id)
    new_session_id = chat_id + "_" + str(int(datetime.datetime.now().timestamp()))
    chat_session_map[chat_id] = new_session_id
    bot.reply_to(message, f"💡 تاریخچه چت ریفرش شد. یک جلسه چت جدید آغاز شد (Session ID: {new_session_id}).")
    logging.info(f"/new_chat command processed for chat {chat_id}. New session ID: {new_session_id}")

@bot.message_handler(commands=['show_sessions'])
def show_sessions(message):
    chat_id = str(message.chat.id)
    logging.info(f"/show_sessions command triggered for chat {chat_id}. Querying saved sessions...")
    try:
        with engine.connect() as connection:
            query = text("SELECT DISTINCT session_id FROM chat_message_histories WHERE session_id LIKE :session_prefix")
            result = connection.execute(query, {"session_prefix": f"{chat_id}%"})
            sessions = [row[0] for row in result.fetchall()]

        if not sessions:
            reply_text = "هیچ جلسه چتی در تاریخچه پیدا نشد."
        else:
            reply_lines = ["جلسات ذخیره شده:"]
            for session_id in sessions:
                if "_" in session_id:
                    parts = session_id.split("_", 1)
                    if len(parts) == 2:
                        try:
                            timestamp = int(parts[1])
                            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception as e:
                            logging.error(f"Failed to convert timestamp in session {session_id}: {e}")
                            date_str = "نامشخص"
                        session_info = f"{session_id} (ایجاد شده در: {date_str})"
                    else:
                        session_info = session_id
                else:
                    session_info = f"{session_id} (جلسه پیش‌فرض)"
                reply_lines.append(session_info)
            reply_text = "\n".join(reply_lines)
        bot.reply_to(message, reply_text)
        logging.info(f"/show_sessions command processed for chat {chat_id}. Sessions found: {sessions}")
    except Exception as e:
        error_message = f"متاسفم، در بازیابی جلسات چت مشکلی پیش آمد.\n\nError: {str(e)}"
        bot.reply_to(message, error_message)
        logging.error(f"Error in /show_sessions for chat {chat_id}: {e}")

# --- Callback Query Handlers for Inline Settings ---
@bot.callback_query_handler(func=lambda call: call.data == "load_business_info")
def handle_load_business_info(call):
    bot.answer_callback_query(call.id)
    logging.info(f"Loading business info for chat {call.message.chat.id}")
    msg = bot.send_message(call.message.chat.id, "لطفاً اطلاعات  کسب و کار و کارکنان خود را وارد کنید:")
    bot.register_next_step_handler(msg, process_business_info)

def process_business_info(message):
    chat_id = str(message.chat.id)
    business_info = message.text
    business_info_map[chat_id] = business_info
    bot.send_message(chat_id, "اطلاعات کسب و کار با موفقیت بارگذاری شد.")
    logging.info(f"Business info loaded for chat {chat_id}: {business_info}")

@bot.callback_query_handler(func=lambda call: call.data == "ai_tone")
def handle_ai_tone(call):
    bot.answer_callback_query(call.id)
    logging.info(f"Setting AI tone for chat {call.message.chat.id}")
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_formal = telebot.types.InlineKeyboardButton("رسمی", callback_data="set_ai_tone_رسمی")
    btn_friendly = telebot.types.InlineKeyboardButton("دوستانه", callback_data="set_ai_tone_دوستانه")
    btn_professional = telebot.types.InlineKeyboardButton("حرفه ای", callback_data="set_ai_tone_حرفه ای")
    keyboard.add(btn_formal, btn_friendly, btn_professional)
    bot.send_message(call.message.chat.id, "لطفاً لحن صحبت هوش مصنوعی را انتخاب کنید:", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data.startswith("set_ai_tone_"))
def set_ai_tone(call):
    tone = call.data.split("set_ai_tone_")[1]
    chat_id = str(call.message.chat.id)
    ai_tone_map[chat_id] = tone
    bot.answer_callback_query(call.id, text=f"لحن صحبت هوش مصنوعی به '{tone}' تغییر یافت.")
    bot.send_message(chat_id, f"لحن هوش مصنوعی شما به '{tone}' تنظیم شد.")
    logging.info(f"AI tone set for chat {chat_id}: {tone}")

# --- Callback Query Handlers for Options Command ---
def process_option_prompt(chat_id, prompt_text):
    """Helper function to send an option prompt to the AI in the current session."""
    logging.info(f"Processing option prompt for chat {chat_id}: {prompt_text}")
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, "🤔 در حال فکر کردن...")
    try:
        ai_response = chain_with_history.invoke(
            {
                "input": prompt_text,
                "ai_tone": ai_tone_map.get(chat_id, "دوستانه"),
                "business_info": business_info_map.get(chat_id, "")
            },
            config={"configurable": {"session_id": chat_id}}
        )
        logging.info(f"AI response for option prompt in chat {chat_id}: {ai_response.content}")
        bot.edit_message_text(ai_response.content, chat_id=chat_id, message_id=placeholder_message.message_id)
    except Exception as e:
        error_message = f"متاسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.\n\nError: {str(e)}"
        bot.edit_message_text(error_message, chat_id=chat_id, message_id=placeholder_message.message_id)
        logging.error(f"Error invoking chain for chat {chat_id} (option prompt): {e}")

@bot.callback_query_handler(func=lambda call: call.data == "option_report_daily")
def handle_option_report_daily(call):
    bot.answer_callback_query(call.id)
    chat_id = str(call.message.chat.id)
    prompt_text = """
    لطفاً کل تاریخچه این چت را به عنوان منبع بررسی در نظر بگیر و فقط پیام‌های مرتبط با فعالیت‌های کاری، تسک‌ها، ورود و خروج و کارهای مرتبط را تحلیل کن. سپس یک گزارش روزانه کامل ارائه بده که شامل موارد زیر باشد:

     **آنالیز تیم:**  
    - بررسی کلی فعالیت‌های کاری تیم  
    - نمایش تسک های امروز تمام تیم به همراه وضعیت هر تسک ، انجام شده یا نشده
    - آنالیز پیشرفت تیم در مسیر پروژه ها
    - تخصیص یک نمره کلی (بین 1 تا 10) به تیم بر اساس عملکرد کلی  
    """
    logging.info(f"Option ' گزارش روزانه تیم' selected for chat {chat_id}.")
    process_option_prompt(chat_id, prompt_text)

@bot.callback_query_handler(func=lambda call: call.data == "option_analyze_today")
def handle_option_analyze_today(call):
    bot.answer_callback_query(call.id)
    chat_id = str(call.message.chat.id)
    prompt_text = "     لطفاً کل تاریخچه این چت را به عنوان منبع بررسی در نظر بگیر و فقط پیام‌های مرتبط با فعالیت‌های کاری، تسک‌ها، ورود و خروج و کارهای مرتبط را تحلیل کن. سپس یک گزارش روزانه کامل برای هر یوزر بنویس و با آنالیز دقیق هر یوزر یه نمره از 1 تا 10 بده و یوزرر هارو بر اساس نمره مرتب کن:"
    logging.info(f"Option 'آنالیز امروز اعضا' selected for chat {chat_id}.")
    process_option_prompt(chat_id, prompt_text)

@bot.callback_query_handler(func=lambda call: call.data == "option_tasks_today")
def handle_option_tasks_today(call):
    bot.answer_callback_query(call.id)
    chat_id = str(call.message.chat.id)
    prompt_text = """
    براساس تاریخچه پیام‌های این چت، برای هر کاربر لیست تسک‌هایی که به انجام آن‌ها اشاره شده است را جداگانه تهیه کن. همچنین، وضعیت هر تسک (انجام شده یا انجام نشده) را مشخص کن.
    """
    logging.info(f"Option 'تسک های امروز' selected for chat {chat_id}.")
    process_option_prompt(chat_id, prompt_text)

# --- Modified Generic Message Handler ---
@bot.message_handler(func=lambda message: message.text is not None and not message.text.startswith("/"))
def handle_message(message):
    """
    پردازش پیام‌های متنی (غیر از دستورات).
    در گروه‌ها، ربات تنها زمانی پاسخ می‌دهد که منشن شود یا کلمه "بلو" در پیام وجود داشته باشد.
    این هندلر از زنجیره LangChain (با تاریخچه پیام) برای تولید پاسخ استفاده می‌کند.
    """
    chat_type = message.chat.type
    chat_id = str(message.chat.id)
    user_message_text = message.text

    sender_first_name = message.from_user.first_name if message.from_user.first_name else message.from_user.username
    modified_user_message = f"اسم کاربر : {sender_first_name}\n{user_message_text}"
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

    if chat_type in ['group', 'supergroup']:
        try:
            history_obj = get_message_history(chat_id)
            try:
                loaded_messages = history_obj.get_messages()
                logging.info(f"{len(loaded_messages)} messages loaded from history for chat {chat_id}.")
            except Exception as e:
                logging.warning(f"Could not load messages count for chat {chat_id}: {e}")
            logging.info(f"Adding new message to history for chat {chat_id}.")
            from langchain_core.messages import HumanMessage
            history_obj.add_message(HumanMessage(content=modified_user_message))
            logging.info(f"New message added to history for chat {chat_id}.")
        except Exception as e:
            logging.error(f"Failed to save group message to history for chat {chat_id}: {e}")

        if not is_mentioned:
            logging.info(f"Bot not mentioned in group chat {chat_id}; no reply will be sent.")
            return

    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.reply_to(message, "🤔 در حال فکر کردن...")

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
        bot.edit_message_text(ai_response.content, chat_id=chat_id, message_id=placeholder_message.message_id)
    except Exception as e:
        error_message = f"متاسفم، مشکلی در پردازش درخواست شما پیش آمد. لطفاً دوباره تلاش کنید.\n\nErroيالr: {str(e)}"
        bot.edit_message_text(error_message, chat_id=chat_id, message_id=placeholder_message.message_id)
        logging.error(f"Error invoking chain for chat {chat_id}: {e}")

def main():
    setup_bot_commands()
    logging.info("🚀 Telegram Bot is running. Chat sessions are managed via LangChain’s history mechanism and saved to SQL as needed...")
    bot.polling(none_stop=True)

if __name__ == "__main__":
    main()
