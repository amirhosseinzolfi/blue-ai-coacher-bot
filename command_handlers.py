"""
command_handlers.py - Handles all Telegram bot commands (/start, /help, etc.)
Part of the Blue AI Coacher Bot system.
"""

import telebot
import logging
import json
import os
from telebot.types import BotCommand, BotCommandScopeDefault, BotCommandScopeAllGroupChats

from utils.helpers import escape_markdown_v2
from config import (
    TELEGRAM_BOT_TOKEN,
    business_info_update_pending
)
from db_manager import save_message_to_history
from langgraph_code import (
    welcome_message,
    help_text_prompt,
    run_agent,
    new_chat_session
)
from daily_reset import scheduler
from prompts.prompts import (
    DAILY_TASK_PROMPT,
    DAILY_REPORT_PROMPT,
    INSTA_IDEA_PROMPT,
    LEADER_BOARD_PROMPT,
    TODAY_COACHING_TIP_PROMPT,
    WELCOME_MESSAGE,
    HELP_TEXT,
    ABOUT_TEXT,
    NEW_CHAT_SESSION_MESSAGE,
    NEW_CHAT_WELCOME_MESSAGE,
    SETTINGS_MENU_PROMPT,
    OPTIONS_MENU_PROMPT,
    IMAGE_GENERATION_USER_PROMPT,
    ERROR_PROCESSING_REQUEST,
    SYSTEM_MESSAGE_SETTINGS_REQUEST,
    SYSTEM_MESSAGE_OPTIONS_REQUEST,
    SYSTEM_MESSAGE_IMAGE_GENERATION_REQUEST,
    SYSTEM_MESSAGE_DAILY_REPORT_REQUEST,
    SYSTEM_MESSAGE_COACHING_TIP_REQUEST,
    SYSTEM_MESSAGE_INSTA_IDEA_REQUEST,
    SYSTEM_MESSAGE_LEADERBOARD_REQUEST,
    # Add missing placeholder constants
    DAILY_REPORT_PLACEHOLDER,
    COACHING_AI_PLACEHOLDER,
    INSTA_IDEA_PLACEHOLDER,
    LEADERBOARD_PLACEHOLDER
)

# Logger will be injected from bot.py
logger = None
# Bot instance will be injected from bot.py
bot = None

def setup_command_handlers(bot_instance, logger_instance):
    """Initialize this module with the bot instance and logger"""
    global bot, logger
    bot = bot_instance
    logger = logger_instance
    setup_bot_commands()
    return {
        'start': send_welcome,
        'help': send_help,
        'about': about_bot,
        'settings': bot_settings,
        'options': options_handler,
        'new_chat': new_chat,
        'generate_image': generate_image_command,
        'menu': send_menu  # Add new menu command
    }

def get_main_menu_keyboard():
    """
    Creates and returns a ReplyKeyboardMarkup for the main menu.
    This keyboard provides quick access buttons for common actions.
    """
    keyboard = telebot.types.ReplyKeyboardMarkup(row_width=3, resize_keyboard=True, one_time_keyboard=False) # Changed row_width to 3
    keyboard.row("➕ افزودن تسک", "📊 گزارش امروز", "💡 کوچینگ با هوش مصنوعی") # Added coaching button to first row
    keyboard.row("🎨 ساخت تصویر", "⚙️ گزینه‌های بیشتر")
    return keyboard

def setup_bot_commands():
    """
    Sets up the bot commands that appear in the Telegram interface.
    """
    commands = [
        BotCommand("start", "شروع ربات و نمایش اطلاعات چت"),
        BotCommand("menu", "نمایش دکمه‌های منو"), # Add menu command to commands list
        BotCommand("options", "نمایش گزینه‌های بیشتر ربات"), # Updated description
        # BotCommand("daily_tasks", "تسک های روزانه"), # Removed as "✅ کار های امروز" button is removed
        BotCommand("chat_report", "گزارش روزانه"),
        BotCommand("insta_idea", "ایده استوری اینستاگرام"),
        BotCommand("generate_image", "ساخت تصویر با هوش مصنوعی"),
        BotCommand("new_chat", "ایجاد جلسه چت جدید"),
        BotCommand("settings", "تنظیمات ربات"),
        BotCommand("about", "اطلاعات ربات"),
        BotCommand("help", "نمایش پیام راهنما")
    ]
    logger.info("Setting up bot commands...")
    try:
        bot.delete_my_commands(scope=BotCommandScopeDefault())
        bot.delete_my_commands(scope=BotCommandScopeAllGroupChats())
    except Exception as e:
        logger.error("Error deleting existing bot commands: %s", e)
    try:
        bot.set_my_commands(commands, scope=BotCommandScopeDefault())
        bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())
        logger.info("Bot commands set successfully.")
    except Exception as e:
        logger.error("Exception during bot command setup: %s", e)

def new_chat(message):
    """
    /new_chat Command: Create a new chat session.
    This command starts a new session while preserving all persistent data.
    """
    chat_id_str = str(message.chat.id)
    try:
        int_chat_id = int(chat_id_str)
    except ValueError:
        logger.error(f"Invalid chat_id in /new_chat: {chat_id_str}")
        bot.reply_to(message, "خطای داخلی رخ داد (شناسه چت نامعتبر).")
        return

    # Create a new chat session.
    new_session_id = new_chat_session(chat_id_str)
    logger.info(f"New chat session created for chat '{int_chat_id}': {new_session_id}. Persistent data is preserved.")

    bot.reply_to(message, escape_markdown_v2(NEW_CHAT_SESSION_MESSAGE), parse_mode="MarkdownV2")
    
    # Add a welcome message to the new session history
    save_message_to_history(chat_id_str, "system", NEW_CHAT_WELCOME_MESSAGE, session_id=new_session_id)

def send_welcome(message):
    """
    /start Command: Send a welcome message, register daily resets and show the main menu keyboard.
    """
    chat_id = str(message.chat.id)
    logger.info("Processing /start command for chat '%s'.", chat_id)
    scheduler.add_chat(chat_id)  # Register chat for daily resets
    main_menu = get_main_menu_keyboard()
    bot.reply_to(message, escape_markdown_v2(WELCOME_MESSAGE), reply_markup=main_menu, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", WELCOME_MESSAGE)

def send_help(message):
    """
    /help Command: Display help information.
    """
    logger.info("Processing /help command.")
    bot.reply_to(message, escape_markdown_v2(HELP_TEXT), parse_mode="MarkdownV2")
    save_message_to_history(str(message.chat.id), "system", HELP_TEXT)

def about_bot(message):
    """
    /about Command: Provide bot information.
    """
    logger.info("Processing /about command for chat '%s'.", message.chat.id)
    bot.reply_to(message, escape_markdown_v2(ABOUT_TEXT), parse_mode="MarkdownV2")
    save_message_to_history(str(message.chat.id), "system", ABOUT_TEXT)

def bot_settings(message):
    """
    /settings Command: Display bot settings options.
    """
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_business_info = telebot.types.InlineKeyboardButton("بارگذاری اطلاعات بیزینس", callback_data="set_business_info")
    btn_ai_tone = telebot.types.InlineKeyboardButton("انتخاب لحن هوش مصنوعی", callback_data="select_ai_tone")
    keyboard.add(btn_business_info, btn_ai_tone)
    logger.info("Processing /settings command for chat '%s'.", chat_id)
    bot.reply_to(message, escape_markdown_v2(SETTINGS_MENU_PROMPT), reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_SETTINGS_REQUEST)

def options_handler(message):
    """
    /options Command: Display additional functions with an attractive inline keyboard.
    """
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_coaching_ai = telebot.types.InlineKeyboardButton("💡 کوچینگ با هوش مصنوعی", callback_data="coaching_with_ai")
    btn_insta_idea = telebot.types.InlineKeyboardButton("✨ ایده اینستا", callback_data="instagram_story_idea")
    btn_leaderboard = telebot.types.InlineKeyboardButton("🏆 رتبه‌بندی", callback_data="leaderboard")
    btn_generate_image = telebot.types.InlineKeyboardButton("🎨 ساخت تصویر", callback_data="generate_image")
    
    keyboard.row(btn_coaching_ai, btn_insta_idea)
    keyboard.row(btn_leaderboard, btn_generate_image)
    
    logger.info("Processing /options command for chat '%s'.", chat_id)
    bot.reply_to(message, escape_markdown_v2(OPTIONS_MENU_PROMPT), reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_OPTIONS_REQUEST)

def generate_image_command(message):
    """
    /generate_image Command: Start the image generation process
    """
    chat_id = str(message.chat.id)
    prompt = bot.reply_to(
        message,
        escape_markdown_v2(IMAGE_GENERATION_USER_PROMPT),
        parse_mode="MarkdownV2"
    )
    logger.info(f"Image generation process started for chat: {chat_id}")
    save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_IMAGE_GENERATION_REQUEST)
    
    # Import the process_image_generation function from message_handlers
    # This is imported here rather than at the top to avoid circular imports
    from message_handlers import process_image_generation
    bot.register_next_step_handler(prompt, process_image_generation)

def load_users_tasks_json() -> str:
    """Load the user tasks JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Corrected path to be relative to the script's directory
    db_path = os.path.join(script_dir, "database", "users_task.json")
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to load users_task.json: {e}")
        return "{}"

def process_main_menu_command(message, command):
    """Process main menu button commands"""
    chat_id = str(message.chat.id)
    user_name = message.from_user.first_name or message.from_user.username
    placeholder_message = None

    try:
        if command == "✅ کار های امروز": # This case will no longer be triggered by main menu
            # This logic might still be useful if /daily_tasks command is kept or called otherwise.
            # For now, assuming it's effectively removed from main menu path.
            # If you want to keep /daily_tasks command functional, this block should remain.
            # However, the prompt implies removing the "today tasks reply keyboard button".
            # To fully remove its effect from this function if not called otherwise:
            logger.info(f"'{command}' button pressed, but it's deprecated from main menu.")
            # bot.reply_to(message, escape_markdown_v2("این گزینه دیگر از طریق منوی اصلی در دسترس نیست."), parse_mode="MarkdownV2")
            # return False # Or handle as appropriate
            # For now, let's assume it's unreachable via menu, so we can effectively remove the block or leave it.
            # To be safe and clear, let's remove the specific handling for "✅ کار های امروز"
            # as it's no longer a main menu button.
            # Remove this entire if block as "✅ کار های امروز" is fully deprecated
            pass 

        elif command == "📊 گزارش امروز":
            placeholder_message = bot.reply_to(message, escape_markdown_v2(DAILY_REPORT_PLACEHOLDER), parse_mode="MarkdownV2")
            save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_DAILY_REPORT_REQUEST)
            response = run_agent(DAILY_REPORT_PROMPT, chat_id, placeholder_message.message_id, user_name)

        elif command == "💡 کوچینگ با هوش مصنوعی":
            placeholder_message = bot.reply_to(message, escape_markdown_v2(COACHING_AI_PLACEHOLDER), parse_mode="MarkdownV2")
            save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_COACHING_TIP_REQUEST)
            response = run_agent(TODAY_COACHING_TIP_PROMPT, chat_id, placeholder_message.message_id, user_name)

        elif command == "✨ ایده اینستا":
            placeholder_message = bot.reply_to(message, escape_markdown_v2(INSTA_IDEA_PLACEHOLDER), parse_mode="MarkdownV2")
            save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_INSTA_IDEA_REQUEST)
            response = run_agent(INSTA_IDEA_PROMPT, chat_id, placeholder_message.message_id, user_name)

        elif command == "🏆 رتبه‌بندی":
            placeholder_message = bot.reply_to(message, escape_markdown_v2(LEADERBOARD_PLACEHOLDER), parse_mode="MarkdownV2")
            # ...existing code for loading tasks...
            save_message_to_history(chat_id, "system", SYSTEM_MESSAGE_LEADERBOARD_REQUEST)
            prompt = LEADER_BOARD_PROMPT.format(users_tasks_json=tasks_json)
            response = run_agent(prompt, chat_id, placeholder_message.message_id, user_name)

        # If we reached here, we have a response and a placeholder to edit
        if placeholder_message and 'response' in locals() and response is not None: # Ensure response is defined
            bot.edit_message_text(escape_markdown_v2(response), chat_id=chat_id, message_id=placeholder_message.message_id, parse_mode="MarkdownV2")
            save_message_to_history(chat_id, "assistant", response)
            return True
        elif command == "✅ کار های امروز": # Explicitly handle removal if needed, or ensure this path isn't taken
             # This path should not be hit if button is removed.
             # If /daily_tasks command still exists and calls this, it needs separate logic.
             # For now, this makes sure it doesn't fall through if accidentally called.
            logger.debug(f"'{command}' was called but is deprecated from main menu.")
            return False


    except Exception as e:
        logger.error(f"[bold red]❌ Error handling menu button '{command}': {e}[/bold red]", exc_info=True)
        error_message = escape_markdown_v2(ERROR_PROCESSING_REQUEST)
        if placeholder_message:
            try:
                bot.edit_message_text(error_message, chat_id=chat_id, message_id=placeholder_message.message_id, parse_mode="MarkdownV2")
            except Exception as edit_error:
                logger.error(f"Failed to edit placeholder message with error: {edit_error}")
                bot.send_message(chat_id, error_message, parse_mode="MarkdownV2")
        else:
            bot.reply_to(message, error_message, parse_mode="MarkdownV2")
    
    return False

def send_menu(message):
    """
    /menu Command: Send just the main menu keyboard without any additional text.
    """
    chat_id = str(message.chat.id)
    logger.info("Processing /menu command for chat '%s'.", chat_id)
    main_menu = get_main_menu_keyboard()
    bot.send_message(message.chat.id, "منوی اصلی:", reply_markup=main_menu)
    save_message_to_history(chat_id, "system", "منوی اصلی نمایش داده شد.")
