#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bot.py – Main Telegram bot interface integrating LangChain, MongoDB,
and G4F API (LLM-based responses).

Note: This file has been refactored for improved readability and structure.
All core functionalities are preserved.
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
import json
import base64
import re
import io

############################################
# Third-Party Imports
############################################
import requests
import PyPDF2
import telebot
import g4f
from telebot.types import BotCommandScopeAllGroupChats, BotCommandScopeDefault, BotCommand

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

# Custom rich logger and helper functions
from utils.rich_logger import setup_logger, display_content, log_function, log_telegram_message, log_api_interaction, log_summarization, log_ai_interaction
from utils.helpers import format_multimodal_input, refine_ai_response, escape_markdown_v2

# Import configuration variables and prompt templates
from config import (
    TELEGRAM_BOT_TOKEN,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    business_info_update_pending,  # For business info updates
    business_info_mode,
    ai_tone_update_pending,
    MONGO_CONNECTION_STRING,
    DATABASE_NAME,
    COLLECTION_NAME,
    BUSINESS_INFO_COLLECTION
)
from prompts.prompts import (
    PROMPT_TEMPLATE_TEXT,
    DAILY_TASK_PROMPT,
    SUMMARY_PROMPT,
    DAILY_REPORT_PROMPT,
    INSTA_IDEA_PROMPT,
    IMAGE_ANALYZER_PROMPT,
    BUSINESS_INFO_SUMMARY_PROMPT,
    WELCOME_MESSAGE,
    HELP_TEXT,
    USER_REPORT_PROMPT,
    SUMMARY_PROMPT_TEXT
)

# Import Telegram Bot Handlers & LangChain Integrations from the central module
from telegram_bot import (
    save_message_to_history,
    get_user_business_info,
    save_user_business_info,
    process_business_info,
    summarize_business_info,
    welcome_message,
    help_text_prompt,
    prompt,
    logger,
    run_agent
)

############################################
# Logger and Event Loop Setup
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
        logger.process_start("[bold cyan]Starting G4F Interference API server on http://localhost:15201/v1 ...[/bold cyan]")
        run_api(bind="0.0.0.0:15201")
    api_thread = threading.Thread(target=start_interference_api, daemon=True, name="G4F-API-Thread")
    api_thread.start()
else:
    logger.warning("[bold orange]G4F API server not started due to missing module.[/bold orange]")

def wait_for_api_server(timeout=30):
    """
    Waits for the G4F API server to become available.
    """
    base_url = "http://localhost:15201/v1/chat/completions"
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
# Initialize Telebot and Set Update Listener
############################################
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
bot.set_update_listener(group_message_listener)

############################################
# Telegram Bot Command Handlers
############################################
def is_admin(chat_id, user_id):
    """
    Checks if the given user is an admin in the chat.
    """
    try:
        member = bot.get_chat_member(chat_id, user_id)
        is_admin_status = member.status in ['creator', 'administrator']
        logger.debug(f"Admin check for user '{user_id}' in chat '{chat_id}': {is_admin_status}")
        return is_admin_status
    except Exception as e:
        logger.error(f"Error checking admin status for user '{user_id}' in chat '{chat_id}': {e}")
        return False

def setup_bot_commands():
    """
    Sets up the bot commands for both private and group chats.
    """
    commands = [
        BotCommand("start", "شروع ربات و نمایش اطلاعات چت"),
        BotCommand("new_chat", "ایجاد جلسه چت جدید"),
        BotCommand("history", "نمایش تاریخچه جلسات"),
        BotCommand("options", "گزینه‌های اضافی"),
        BotCommand("help", "نمایش پیام راهنما"),
        BotCommand("settings", "تنظیمات ربات"),
        BotCommand("about", "اطلاعات ربات")
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

@bot.message_handler(commands=['new_chat'])
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

    from telegram_bot import new_chat_session, save_message_to_history

    # Create a new chat session.
    new_session_id = new_chat_session(chat_id_str)
    logger.info(f"New chat session created for chat '{int_chat_id}': {new_session_id}. Persistent data is preserved.")

    response_text = "🆕 جلسه چت جدید ایجاد شد. تاریخچه چت جدید آغاز گردید. اطلاعات بیزینس قبلی حفظ شده‌اند."
    bot.reply_to(message, escape_markdown_v2(response_text), parse_mode="MarkdownV2")
    
    # Add a welcome message to the new session history (saved with the new session ID).
    welcome_text = "جلسه گفتگوی جدید آغاز شد. چطور می‌توانم به شما کمک کنم؟"
    save_message_to_history(chat_id_str, "system", welcome_text, session_id=new_session_id)
@bot.message_handler(commands=['start'])
def send_welcome(message):
    """
    /start Command: Send a welcome message.
    """
    chat_id = str(message.chat.id)
    logger.info("Processing /start command for chat '%s'.", chat_id)
    bot.reply_to(message, escape_markdown_v2(welcome_message), parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", welcome_message)

@bot.message_handler(commands=['help'])
def send_help(message):
    """
    /help Command: Display help information.
    """
    logger.info("Processing /help command.")
    bot.reply_to(message, escape_markdown_v2(help_text_prompt), parse_mode="MarkdownV2")
    save_message_to_history(str(message.chat.id), "system", help_text_prompt)

@bot.message_handler(commands=['about'])
def about_bot(message):
    """
    /about Command: Provide bot information.
    """
    about_text = (
        "🤖 *درباره ربات:*\n\n"
        "من **بلو** هستم، مربی کسب‌وکار هوشمند با پشتیبانی از فناوری LangChain و مدل‌های OpenAI.\n"
        "برای اطلاعات بیشتر از دستور `/help` استفاده کنید."
    )
    logger.info("Processing /about command for chat '%s'.", message.chat.id)
    bot.reply_to(message, escape_markdown_v2(about_text), parse_mode="MarkdownV2")
    save_message_to_history(str(message.chat.id), "system", about_text)

@bot.message_handler(commands=['settings'])
def bot_settings(message):
    """
    /settings Command: Display bot settings options.
    """
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_business_info = telebot.types.InlineKeyboardButton("بارگذاری اطلاعات بیزینس", callback_data="load_business_info")
    btn_ai_tone = telebot.types.InlineKeyboardButton("انتخاب لحن هوش مصنوعی", callback_data="ai_tone")
    keyboard.add(btn_business_info, btn_ai_tone)
    settings_text = "⚙️ *تنظیمات ربات:*\n\nلطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
    logger.info("Processing /settings command for chat '%s'.", chat_id)
    bot.reply_to(message, settings_text, reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", settings_text)

@bot.message_handler(commands=['options'])
def options_handler(message):
    """
    /options Command: Display additional functions.
    """
    chat_id = str(message.chat.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    btn_daily_tasks = telebot.types.InlineKeyboardButton("تسک های روزانه", callback_data="daily_tasks")
    btn_instagram_story = telebot.types.InlineKeyboardButton("ایده استوری اینستاگرام", callback_data="instagram_story_idea")
    btn_chat_report = telebot.types.InlineKeyboardButton("گزارش روزانه", callback_data="chat_report")
    btn_shamsi_date = telebot.types.InlineKeyboardButton("تاریخ شمسی امروز", callback_data="shamsi_date")
    keyboard.row(btn_daily_tasks, btn_instagram_story, btn_chat_report)
    keyboard.row(btn_shamsi_date)
    options_text = "⚙️ *انتخاب گزینه‌ها:*\nلطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
    logger.info("Processing /options command for chat '%s'.", chat_id)
    bot.reply_to(message, options_text, reply_markup=keyboard, parse_mode="MarkdownV2")
    save_message_to_history(chat_id, "system", options_text)

############################################
# Callback Handlers for Inline Buttons
############################################
@bot.callback_query_handler(func=lambda call: call.data == "daily_tasks")
def handle_daily_tasks(call):
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logger.info(f"[bold cyan]🔘 Daily Tasks button clicked by {sender_first_name} in chat {chat_id}[/bold cyan]")
    
    prompt_input = DAILY_TASK_PROMPT
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش وظایف روزانه..."), parse_mode="MarkdownV2")
    
    try:
        refined_response = run_agent(f"{sender_first_name} : {prompt_input}", chat_id, placeholder_message.message_id, sender_first_name)
        save_message_to_history(chat_id, "assistant", refined_response)
        logger.info(f"[bold green]✅ Daily task response sent to {sender_first_name}[/bold green]")
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش وظایف روزانه پیش آمد. لطفاً دوباره تلاش کنید.")
        logger.error(f"[bold red]❌ Error in Daily Tasks handler: {str(e)}[/bold red]")
        bot.edit_message_text(error_message, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")

@bot.callback_query_handler(func=lambda call: call.data == "instagram_story_idea")
def handle_instagram_story_idea(call):
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logger.info(f"[bold cyan]🔘 Instagram Story idea button clicked by {sender_first_name} in chat {chat_id}[/bold cyan]")
    
    prompt_input = INSTA_IDEA_PROMPT
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش ایده استوری اینستاگرام..."), parse_mode="MarkdownV2")
    
    try:
        refined_response = run_agent(f"{sender_first_name} : {prompt_input}", chat_id, placeholder_message.message_id, sender_first_name)
        save_message_to_history(chat_id, "assistant", refined_response)
        logger.info(f"[bold green]✅ Instagram story idea sent to {sender_first_name}[/bold green]")
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش ایده استوری اینستاگرام پیش آمد. لطفاً دوباره تلاش کنید.")
        logger.error("Error in Instagram Story Idea handler for chat '%s': %s", chat_id, e)
        bot.edit_message_text(error_message, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")

@bot.callback_query_handler(func=lambda call: call.data == "chat_report")
def handle_chat_report(call):
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logger.info(f"[bold cyan]🔘 Chat Report button clicked by {sender_first_name} in chat {chat_id}[/bold cyan]")
    
    prompt_input = DAILY_REPORT_PROMPT
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش گزارش تاریخچه چت کاربران..."), parse_mode="MarkdownV2")
    
    try:
        refined_response = run_agent(f"{sender_first_name} : {prompt_input}", chat_id, placeholder_message.message_id, sender_first_name)
        save_message_to_history(chat_id, "assistant", refined_response)
        logger.info(f"[bold green]✅ Chat report sent to {sender_first_name}[/bold green]")
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش گزارش تاریخچه چت کاربران پیش آمد. لطفاً دوباره تلاش کنید.")
        logger.error("Error in Chat Report handler for chat '%s': %s", chat_id, e)
        bot.edit_message_text(error_message, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")

@bot.callback_query_handler(func=lambda call: call.data == "shamsi_date")
def handle_shamsi_date(call):
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logger.info(f"[bold cyan]🔘 Shamsi Date button clicked by {sender_first_name} in chat {chat_id}[/bold cyan]")
    
    from utils.date_helpers import get_full_shamsi_date
    shamsi_date = get_full_shamsi_date()
    
    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال پردازش تاریخ شمسی امروز..."), parse_mode="MarkdownV2")
    
    try:
        prompt_input = f"تاریخ امروز در تقویم شمسی ایران {shamsi_date} است. لطفاً این تاریخ را در نظر بگیر و به من بگو که امروز چه روزی است و نکات مناسبتی و یا فصلی مرتبط با این تاریخ را توضیح بده."
        refined_response = run_agent(f"{sender_first_name} : {prompt_input}", chat_id, placeholder_message.message_id, sender_first_name)
        save_message_to_history(chat_id, "assistant", refined_response)
        logger.info(f"[bold green]✅ Shamsi date response sent to {sender_first_name}[/bold green]")
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
    except Exception as e:
        error_message = escape_markdown_v2(f"❌ متأسفم، مشکلی در پردازش تاریخ شمسی پیش آمد. تاریخ شمسی امروز: {shamsi_date}")
        logger.error(f"[bold red]❌ Error in Shamsi Date handler: {str(e)}[/bold red]")
        bot.edit_message_text(error_message, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")

@bot.callback_query_handler(func=lambda call: call.data == "load_business_info")
def handle_load_business_info(call):
    chat_id = str(call.message.chat.id)
    current_info = get_user_business_info(chat_id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    if current_info:
        btn_replace = telebot.types.InlineKeyboardButton("تعویض اطلاعات", callback_data="replace_business_info")
        btn_append = telebot.types.InlineKeyboardButton("افزودن به اطلاعات", callback_data="append_business_info")
        keyboard.add(btn_replace, btn_append)
        prompt_text = f"اطلاعات بیزینس فعلی:\n{current_info}\n\nآیا می‌خواهید آن را تعویض کنید یا اطلاعات جدید به آن اضافه شود؟"
        bot.answer_callback_query(call.id)
        bot.send_message(chat_id, prompt_text, reply_markup=keyboard)
    else:
        business_info_update_pending[chat_id] = True
        business_info_mode[chat_id] = "replace"
        bot.answer_callback_query(call.id, "لطفاً اطلاعات بیزینس خود را به عنوان پیام ارسال کنید.")
        bot.send_message(chat_id, "📄 لطفاً اطلاعات بیزینس خود را ارسال کنید:")

@bot.callback_query_handler(func=lambda call: call.data == "replace_business_info")
def handle_replace_business_info(call):
    chat_id = str(call.message.chat.id)
    business_info_update_pending[chat_id] = True
    business_info_mode[chat_id] = "replace"
    bot.answer_callback_query(call.id, "لطفاً اطلاعات جدید بیزینس را ارسال کنید تا جایگزین اطلاعات فعلی شود.")
    bot.send_message(chat_id, "📄 اطلاعات جدید بیزینس خود را ارسال کنید:")

@bot.callback_query_handler(func=lambda call: call.data == "append_business_info")
def handle_append_business_info(call):
    chat_id = str(call.message.chat.id)
    business_info_update_pending[chat_id] = True
    business_info_mode[chat_id] = "append"
    current_info = get_user_business_info(chat_id)
    bot.answer_callback_query(call.id, "لطفاً اطلاعات جدید بیزینس را ارسال کنید تا به اطلاعات فعلی اضافه شود.")
    bot.send_message(chat_id, f"📄 اطلاعات فعلی بیزینس:\n{current_info}\n\nاطلاعات جدید را ارسال کنید:")

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

@bot.callback_query_handler(func=lambda call: call.data in ["ai_tone_dostane", "ai_tone_rasmi", "ai_tone_pro"])
def select_ai_tone(call):
    chat_id = str(call.message.chat.id)
    tone_mapping = {
        "ai_tone_dostane": ("دوستانه", "friendly , cool and kind"),
        "ai_tone_rasmi": ("رسمی", "official , serioous and formal"),
        "ai_tone_pro": ("حرفه ای و پروفشنال", "professional , expert and business-like"),
    }
    selected = tone_mapping.get(call.data)
    if selected:
        from config import ai_tone_map
        ai_tone_map[chat_id] = f"{selected[0]}: {selected[1]}"
        bot.answer_callback_query(call.id, f"لحن انتخاب شده: {selected[0]}")
        bot.send_message(chat_id, f"✅ لحن هوش مصنوعی به '{selected[0]}' تغییر یافت.")

############################################
# Content Handlers for Text, Photo, and Document
############################################
@bot.message_handler(content_types=['text', 'photo'])
@log_function(logger)
def handle_message(message):
    chat_type = message.chat.type
    chat_id = str(message.chat.id)
    sender_first_name = message.from_user.first_name or message.from_user.username

    logger.info("[bold blue]" + "-"*40 + "[/bold blue]")
    logger.info(f"[blue]💬 New message from {sender_first_name} in {chat_type} {chat_id}[/blue]")
    
    if message.content_type == 'photo':
        text_component = message.caption if message.caption else ""
        logger.info(f"[magenta]📷 Photo received with caption: {text_component[:50]}...[/magenta]")
    else:
        logger.info(f"[cyan]📝 Text: {message.text[:50]}...[/cyan]")

    # Check for AI Tone update or Business Info update messages
    if ai_tone_update_pending.get(chat_id):
        new_tone = message.text.strip() if message.text else ""
        from config import ai_tone_map
        ai_tone_map[chat_id] = new_tone
        ai_tone_update_pending.pop(chat_id, None)
        bot.send_message(chat_id, f"✅ لحن هوش مصنوعی به '{new_tone}' تغییر یافت.")
        return
    if business_info_update_pending.get(chat_id):
        new_info = process_business_info(message.text if message.text else "", chat_id)
        mode = business_info_mode.get(chat_id, "replace")
        if mode == "append":
            current_info = get_user_business_info(chat_id)
            optimized = summarize_business_info(current_info + "\n" + new_info)
        else:
            optimized = summarize_business_info(new_info)
        save_user_business_info(chat_id, optimized)
        business_info_update_pending.pop(chat_id, None)
        business_info_mode.pop(chat_id, None)
        bot.send_message(chat_id, "✅ اطلاعات بیزینس به‌روزرسانی شد.")
        bot.send_message(chat_id, f"🔍 اطلاعات بیزینس نهایی:\n{optimized}")
        return

    # Process Photo vs. Text Message
    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        text_component = message.caption if message.caption else ""
        query_payload = []
        if text_component:
            query_payload.append({"type": "text", "text": text_component})
        query_payload.append({"type": "image_url", "image_url": {"url": file_url}})
        user_message_text = text_component if text_component else "تصویر دریافت شد."
        logger.info(f"[bold cyan]📷 Photo received from {sender_first_name} in chat {chat_id} with caption: {text_component}[/bold cyan]")
    else:
        query_payload = message.text
        user_message_text = message.text

    logger.info(f"[bold cyan]💬 Message from {sender_first_name} in chat {chat_id}: {user_message_text[:50]}{'...' if len(user_message_text) > 50 else ''}[/bold cyan]")
    save_message_to_history(chat_id, "user", f"{sender_first_name}: {user_message_text}")

    # In group chats, respond only when the bot is mentioned
    if chat_type in ['group', 'supergroup']:
        bot_username = bot.get_me().username
        is_mentioned = (
            (message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id) or
            (user_message_text and (bot_username in user_message_text)) or
            (message.entities and any(
                entity.type == 'mention' and 
                user_message_text[entity.offset:entity.offset + entity.length].lower() == f"@{bot_username.lower()}"
                for entity in message.entities
            )) or ("بلو" in user_message_text)
        )
        if not is_mentioned:
            logger.debug(f"[dim]Message in group {chat_id} not mentioning bot - no response[/dim]")
            return

    bot.send_chat_action(chat_id, 'typing')
    placeholder_message = bot.reply_to(message, escape_markdown_v2("🤔 در حال فکر کردن..."), parse_mode="MarkdownV2")
    
    try:
        refined_response = run_agent(query_payload, chat_id, placeholder_message.message_id, sender_first_name)
        logger.info(f"[green]✅ Response sent ({len(refined_response)} chars)[/green]")
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
        # Save message to history (no background summarization call)
        save_message_to_history(chat_id, "assistant", refined_response)
        
    except Exception as e:
        logger.error(f"[bold red]❌ Error: {str(e)}[/bold red]")
        error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش درخواست شما پیش آمد.")
        bot.edit_message_text(error_message, chat_id=chat_id,
                          message_id=placeholder_message.message_id,
                          parse_mode="MarkdownV2")
    logger.info("[bold blue]" + "-"*40 + "[/bold blue]")

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
        save_user_business_info(chat_id, optimized)
        business_info_update_pending.pop(chat_id, None)
        business_info_mode.pop(chat_id, None)
        bot.send_message(chat_id, "✅ اطلاعات بیزینس به‌روزرسانی شد.")
        bot.send_message(chat_id, f"🔍 اطلاعات بیزینس نهایی:\n{optimized}")
    else:
        bot.reply_to(message, "دستور به‌روزرسانی اطلاعات بیزینس فعال نیست.")

############################################
# Connection Management
############################################
import aiohttp
from contextlib import suppress

async def cleanup_aiohttp_sessions():
    """Clean up any remaining aiohttp sessions"""
    # Clean up g4f client session
    if hasattr(client, 'session') and isinstance(client.session, aiohttp.ClientSession):
        if not client.session.closed:
            await client.session.close()
            # Give time for cleanup
            await asyncio.sleep(0.25)
    
    # Clean up any pending connections
    pending = asyncio.all_tasks()
    for task in pending:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

def cleanup():
    """Cleanup function to be called on shutdown"""
    from db_manager import db_manager
    
    # Close MongoDB connection
    db_manager.close()
    
    # Clean up aiohttp sessions
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(cleanup_aiohttp_sessions())
    else:
        loop.run_until_complete(cleanup_aiohttp_sessions())

# Register cleanup for both normal exit and signals
import signal
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
        setup_bot_commands()
        logger.info("Bot commands configured")
        logger.info("Bot is running...")
        logger.info("=" * 50)
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    main()
