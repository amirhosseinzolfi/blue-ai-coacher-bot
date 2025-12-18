#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bot.py – Main Telegram bot interface integrating LangChain, 
G4F API and AI components.

This is the main entry point for the Blue AI Coacher Bot system.
It handles bot initialization, G4F API server setup, and coordination 
of the modular components.
"""

############################################
# Standard Library Imports
############################################
import os
import datetime
import logging
import threading
import atexit
import asyncio
import time
import signal

############################################
# Third-Party Imports
############################################
import requests
import telebot
import g4f

# Disable SSL verification for testing (DO NOT use in production)
import telebot.apihelper as apihelper
apihelper.SESSION = requests.Session()
apihelper.SESSION.verify = False

############################################
# Local Imports and Custom Module Setup
############################################
# Patch g4f's AuthResult for missing api_key property if needed
try:
    from g4f.Provider.needs_auth.OpenaiChat import AuthResult
    if not hasattr(AuthResult, "api_key"):
        def _get_api_key(self):
            return getattr(self, "key", None)
        AuthResult.api_key = property(_get_api_key)
except Exception as e:
    print("Failed to patch AuthResult:", e)

# Client for G4F
from g4f.client import Client
client = Client()

# Custom rich logger setup
from utils.rich_logger import setup_logger, display_content, log_function, log_telegram_message
from utils.helpers import escape_markdown_v2

# Import configuration and prompt templates
from config import (
    TELEGRAM_BOT_TOKEN,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    business_info_update_pending
)

# Import modules for database operations
from db_manager import save_message_to_history
from prompts.prompts import TASK_ENTRY_PROMPT

# Import daily reset scheduler
from daily_reset import scheduler

############################################
# Logger Setup
############################################
logger = setup_logger(level=logging.INFO)
logger.info("[bold blue]Blue Business Bot Starting[/bold blue]")

# For Windows: avoid asyncio warnings
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

############################################
# G4F API Server Setup
############################################
try:
    from g4f.api import run_api
except ImportError:
    logger.error("[bold red]g4f.api module not found. Install the 'g4f' package.[/bold red]")
    run_api = None

if run_api is not None:
    def start_interference_api():
        logger.process_start("[bold cyan]Starting G4F Interference API server on http://localhost:15501/v1 ...[/bold cyan]")
        run_api(bind="0.0.0.0:15501")
    api_thread = threading.Thread(target=start_interference_api, daemon=True, name="G4F-API-Thread")
    api_thread.start()
else:
    logger.warning("[bold orange]G4F API server not started due to missing module.[/bold orange]")

def wait_for_api_server(timeout=30):
    """
    Waits for the G4F API server to become available.
    """
    base_url = "http://localhost:15501/v1/chat/completions"
    start_time = datetime.datetime.now()
    logger.process_start("[bold yellow]Waiting for the G4F API server to become available...[/bold yellow]")
    while True:
        try:
            response = requests.post(base_url, json={"messages": [{"role": "system", "content": "ping"}]}, timeout=5)
            if response.ok:
                logger.process_end("[bold green]G4F API server responded successfully.[/bold green]")
                break
        except Exception:
            pass
        if (datetime.datetime.now() - start_time).seconds > timeout:
            logger.error("[bold red]API server not available after waiting 30 seconds.[/bold red]")
            return
        time.sleep(1)

wait_for_api_server()

############################################
# Group Message Listener (for Group Chats)
############################################
def group_message_listener(messages):
    for message in messages:
        if message.chat.type in ['group', 'supergroup']:
            chat_id = str(message.chat.id)
            sender = message.from_user.first_name or message.from_user.username
            # Process non-text messages
            if message.content_type != "text":
                if message.content_type == "photo":
                    text = f"{sender} sent a photo."
                    save_message_to_history(chat_id, "user", text)
                    log_telegram_message(logger, message, "received")
                elif message.content_type in ['video', 'audio', 'document']:
                    text = f"{sender} sent a {message.content_type}."
                    save_message_to_history(chat_id, "user", text)
                    log_telegram_message(logger, message, "received")

############################################
# Initialize Telebot and Load Handlers
############################################
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
bot.set_update_listener(group_message_listener)

# Import and setup command handlers
from command_handlers import setup_command_handlers
command_handlers = setup_command_handlers(bot, logger)

# Import and setup callback handlers
from callback_handlers import setup_callback_handlers
callback_handlers = setup_callback_handlers(bot, logger)

# Import and setup message handlers
from message_handlers import setup_message_handlers
message_handlers = setup_message_handlers(bot, logger)

############################################
# Register Command Handlers
############################################
@bot.message_handler(commands=['start'])
def start_command(message):
    command_handlers['start'](message)

@bot.message_handler(commands=['help'])
def help_command(message):
    command_handlers['help'](message)

@bot.message_handler(commands=['menu'])
def menu_command(message):
    command_handlers['menu'](message)

@bot.message_handler(commands=['about'])
def about_command(message):
    command_handlers['about'](message)

@bot.message_handler(commands=['settings'])
def settings_command(message):
    command_handlers['settings'](message)

@bot.message_handler(commands=['options'])
def options_command(message):
    command_handlers['options'](message)

@bot.message_handler(commands=['new_chat'])
def new_chat_command(message):
    command_handlers['new_chat'](message)

@bot.message_handler(commands=['generate_image'])
def generate_image_command(message):
    command_handlers['generate_image'](message)

@bot.message_handler(commands=['clear_data'])
def clear_data_command(message):
    command_handlers['clear_data'](message)

@bot.message_handler(commands=['session_assistant'])
def session_assistant_command(message):
    command_handlers['session_assistant'](message)

############################################
# Register Callback Query Handlers
############################################
@bot.callback_query_handler(func=lambda call: call.data == "set_business_info")
def set_business_info_callback(call):
    callback_handlers['set_business_info'](call)

@bot.callback_query_handler(func=lambda call: call.data == "select_ai_tone")
def select_ai_tone_callback(call):
    callback_handlers['select_ai_tone'](call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("set_tone_"))
def set_tone_selection_callback(call):
    callback_handlers['set_tone_'](call)

@bot.callback_query_handler(func=lambda call: call.data == "generate_image")
def generate_image_callback(call):
    callback_handlers['generate_image'](call)

# Added new callback handlers
@bot.callback_query_handler(func=lambda call: call.data == "coaching_with_ai")
def coaching_with_ai_callback(call):
    callback_handlers['coaching_with_ai'](call)

@bot.callback_query_handler(func=lambda call: call.data == "instagram_story_idea")
def instagram_story_idea_callback(call):
    callback_handlers['instagram_story_idea'](call)

@bot.callback_query_handler(func=lambda call: call.data == "leaderboard")
def leaderboard_callback(call):
    callback_handlers['leaderboard'](call)

@bot.callback_query_handler(func=lambda call: call.data == "confirm_clear_data")
def confirm_clear_data_callback(call):
    callback_handlers['confirm_clear_data'](call)

@bot.callback_query_handler(func=lambda call: call.data == "cancel_clear_data")
def cancel_clear_data_callback(call):
    callback_handlers['cancel_clear_data'](call)

@bot.callback_query_handler(func=lambda call: call.data == "session_assistant")
def session_assistant_callback(call):
    callback_handlers['session_assistant'](call)

############################################
# Register Menu Button Handler
############################################
@bot.message_handler(func=lambda m: m.text in [
    "➕ افزودن تسک", "📊 گزارش امروز", "💡 کوچینگ با هوش مصنوعی", "⚙️ گزینه‌های بیشتر"
])
def main_menu_handler(message):
    """Handle main menu button clicks"""
    if message.text == "➕ افزودن تسک": # Changed "➕ افزودن کار امروز" to "➕ افزودن تسک"
        # Special handling for task entry
        msg = bot.reply_to(message, escape_markdown_v2(TASK_ENTRY_PROMPT), parse_mode="MarkdownV2")
        save_message_to_history(str(message.chat.id), "system", TASK_ENTRY_PROMPT)
        bot.register_next_step_handler(msg, message_handlers['task_entry'])
        return

    elif message.text == "🎨 ساخت تصویر":
        # Direct to image generation command handler
        command_handlers['generate_image'](message)
        return
        
    elif message.text == "⚙️ گزینه‌های بیشتر":
        # Direct to options command handler
        command_handlers['options'](message)
        return

    elif message.text in ["📊 گزارش امروز", "💡 کوچینگ با هوش مصنوعی"]:  # Handle both report and coaching
        # For both "📊 گزارش امروز" and "💡 کوچینگ با هوش مصنوعی"
        from command_handlers import process_main_menu_command
        process_main_menu_command(message, message.text)
        return

############################################
# Register Message Handlers
############################################
@bot.message_handler(content_types=['text', 'photo', 'voice', 'audio'])
@log_function(logger)
def handle_message_wrapper(message):
    message_handlers['text_photo'](message)

@bot.message_handler(content_types=["document"])
def handle_document_wrapper(message):
    message_handlers['document'](message)

############################################
# Connection Management
############################################
import aiohttp
from contextlib import suppress

async def cleanup_aiohttp_sessions():
    """Clean up any remaining aiohttp sessions"""
    if hasattr(client, 'session') and isinstance(client.session, aiohttp.ClientSession):
        if not client.session.closed:
            await client.session.close()
            await asyncio.sleep(0.25)
    
    pending = asyncio.all_tasks()
    for task in pending:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

def cleanup():
    """Cleanup function to be called on shutdown"""
    from db_manager import db_manager
    
    db_manager.close()
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(cleanup_aiohttp_sessions())
    else:
        loop.run_until_complete(cleanup_aiohttp_sessions())

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: cleanup())
signal.signal(signal.SIGTERM, lambda s, f: cleanup())

############################################
# Main Bot Execution
############################################
def main():
    try:
        logger.info("=" * 50)
        logger.info("Starting Blue Business Bot")
        # Setup commands via command_handlers module
        logger.info("Bot commands configured")
        
        scheduler.start()
        logger.info("Daily reset scheduler initialized")
        
        logger.info("Bot is running...")
        logger.info("=" * 50)
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        scheduler.stop()
        cleanup()

if __name__ == "__main__":
    main()
