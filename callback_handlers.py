"""
callback_handlers.py - Handles all inline button callbacks for the Telegram bot
Part of the Blue AI Coacher Bot system.
"""

import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import json # Added
import os   # Added

from utils.helpers import escape_markdown_v2
from config import (
    business_info_update_pending,
    TELEGRAM_BOT_TOKEN
)
from db_manager import save_ai_tone, get_ai_tone, save_message_to_history
# Added imports for new handlers
from langgraph_code import run_agent 
from prompts.prompts import (
    TODAY_COACHING_TIP_PROMPT,
    INSTA_IDEA_PROMPT,
    LEADER_BOARD_PROMPT
)

# Logger will be injected from bot.py
logger = None
# Bot instance will be injected from bot.py
bot = None

def setup_callback_handlers(bot_instance, logger_instance):
    """Initialize this module with the bot instance and logger"""
    global bot, logger
    bot = bot_instance
    logger = logger_instance
    
    # Return a dictionary mapping callback data patterns to handler functions
    return {
        'set_business_info': handle_set_business_info,
        'select_ai_tone': handle_select_ai_tone,
        'set_tone_': handle_set_tone_selection,
        'generate_image': handle_generate_image,
        # Added new handlers
        'coaching_with_ai': handle_coaching_with_ai,
        'instagram_story_idea': handle_instagram_story_idea,
        'leaderboard': handle_leaderboard,
    }

def handle_set_business_info(call):
    """Handles the 'Load Business Info' button click."""
    chat_id = str(call.message.chat.id)
    user_id = str(call.from_user.id)
    logger.info(f"Processing 'set_business_info' callback for chat {chat_id}")
    bot.answer_callback_query(call.id)

    business_info_update_pending[chat_id] = True

    # Updated prompt to mention supported file formats and batch processing
    from data_extractors import get_supported_file_types
    supported_types = get_supported_file_types()
    types_text = "، ".join(supported_types.values())
    
    prompt_text = f"""📋 لطفاً اطلاعات کسب و کار خود را ارسال کنید:

📝 متن مستقیم: تایپ کردن اطلاعات
📁 فایل‌ها: ارسال یک یا چند فایل با فرمت‌های زیر:
• ({types_text})

ربات به طور خودکار متن را از تمام فایل‌ها استخراج کرده و با هوش مصنوعی تحلیل می‌کند."""

    try:
        bot.edit_message_text(escape_markdown_v2(prompt_text), chat_id=chat_id, message_id=call.message.message_id, parse_mode="MarkdownV2")
    except Exception as e:
        logger.warning(f"Could not edit settings message for business info prompt: {e}. Sending new message.")
        bot.send_message(chat_id, escape_markdown_v2(prompt_text), parse_mode="MarkdownV2")

    save_message_to_history(chat_id, "system", "درخواست بروزرسانی اطلاعات کسب و کار")

def handle_select_ai_tone(call):
    """Handles the 'Select AI Tone' button click."""
    chat_id = str(call.message.chat.id)
    chat_type = call.message.chat.type
    logger.info(f"Processing 'select_ai_tone' callback for chat {chat_id}")
    bot.answer_callback_query(call.id)

    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("دوستانه 😊", callback_data="set_tone_friendly"))
    keyboard.add(InlineKeyboardButton("حرفه‌ای 🧐", callback_data="set_tone_professional"))
    keyboard.add(InlineKeyboardButton("خلاق ✨", callback_data="set_tone_creative"))

    current_tone = get_ai_tone(chat_id, chat_type)
    prompt_text = f"لحن فعلی: *{current_tone}*\n\nلطفاً لحن مورد نظر برای پاسخ‌های ربات را انتخاب کنید:"
    try:
        bot.edit_message_text(escape_markdown_v2(prompt_text), chat_id=chat_id, message_id=call.message.message_id, reply_markup=keyboard, parse_mode="MarkdownV2")
    except Exception as e:
        logger.warning(f"Could not edit settings message for AI tone selection: {e}. Sending new message.")
        bot.send_message(chat_id, escape_markdown_v2(prompt_text), reply_markup=keyboard, parse_mode="MarkdownV2")

    save_message_to_history(chat_id, "system", "درخواست تغییر لحن AI")

def handle_set_tone_selection(call):
    """Handles the selection of a specific AI tone."""
    chat_id = str(call.message.chat.id)
    chat_type = call.message.chat.type
    selected_tone_key = call.data.split("set_tone_")[1]
    logger.info(f"Processing tone selection '{selected_tone_key}' for chat {chat_id}")
    bot.answer_callback_query(call.id)

    tone_map = {
        "friendly": "دوستانه",
        "professional": "حرفه‌ای",
        "creative": "خلاق"
    }

    if selected_tone_key in tone_map:
        selected_tone = tone_map[selected_tone_key]
        save_ai_tone(chat_id, selected_tone, chat_type)
        response_text = f"✅ لحن ربات به *{selected_tone}* تغییر یافت."
        bot.edit_message_text(escape_markdown_v2(response_text), chat_id=chat_id, message_id=call.message.message_id, parse_mode="MarkdownV2")
        save_message_to_history(chat_id, "system", f"لحن AI به {selected_tone} تغییر یافت.")
    else:
        bot.edit_message_text(escape_markdown_v2("❌ انتخاب نامعتبر."), chat_id=chat_id, message_id=call.message.message_id, parse_mode="MarkdownV2")

def handle_generate_image(call):
    """
    Handle the generate image button click
    """
    chat_id = str(call.message.chat.id)
    sender_first_name = call.from_user.first_name or call.from_user.username
    logger.info(f"[bold cyan]🔘 Generate Image button clicked by {sender_first_name} in chat {chat_id}[/bold cyan]")
    
    bot.answer_callback_query(call.id)
    prompt = bot.send_message(
        chat_id,
        escape_markdown_v2(
            "🎨 *ساخت تصویر با هوش مصنوعی*\n\n"
            "لطفاً توضیح دهید چه تصویری می‌خواهید بسازم؟ "
            "(مثال: نقاشی آبرنگ از یک گل رز قرمز با پس‌زمینه تاریک)"
        ),
        parse_mode="MarkdownV2"
    )
    save_message_to_history(chat_id, "system", "درخواست ساخت تصویر با هوش مصنوعی")
    
    # Import the process_image_generation function from message_handlers
    # This is imported here rather than at the top to avoid circular imports
    from message_handlers import process_image_generation
    bot.register_next_step_handler(prompt, process_image_generation)

# Helper function for processing similar callback actions
def _process_action_callback(call, prompt_template, action_description, is_leaderboard=False):
    chat_id = str(call.message.chat.id)
    user_name = call.from_user.first_name or call.from_user.username
    message_id = call.message.message_id
    
    bot.answer_callback_query(call.id)

    try:
        bot.edit_message_text(
            escape_markdown_v2("⏳ در حال پردازش، لطفا صبر کنید..."),
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=None,  # Remove inline keyboard
            parse_mode="MarkdownV2"
        )
    except Exception as e:
        logger.warning(f"Could not edit message text for {action_description} status: {e}. Proceeding to run agent.")

    try:
        log_message = f"{user_name}: [انتخاب از گزینه‌ها - {action_description}]"
        save_message_to_history(chat_id, "user", log_message)
        
        current_prompt = prompt_template
        if is_leaderboard:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Corrected path to be relative to the script's directory
            db_path = os.path.join(script_dir, "database", "users_task.json")
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
                chat_tasks = all_data.get(chat_id, {})
                tasks_json_string = json.dumps(chat_tasks, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to load users_task.json for leaderboard: {e}")
                tasks_json_string = "{}"
            current_prompt = prompt_template.format(users_tasks_json=tasks_json_string)

        # run_agent is expected to return the refined response text
        # The placeholder_message_id (message_id here) is for run_agent to know which message context it's in,
        # but the actual editing of this message with the final response is done here.
        # However, run_agent in message_handlers.py and command_handlers.py seems to handle its own placeholder.
        # Let's assume run_agent takes message_id as a context or placeholder to update.
        # For consistency with other uses of run_agent, it should return the response.
        response = run_agent(current_prompt, chat_id, message_id, user_name)
        
        bot.edit_message_text(
            response, # Assumes response is already MarkdownV2 escaped by refine_ai_response via run_agent
            chat_id=chat_id,
            message_id=message_id,
            parse_mode="MarkdownV2"
        )
        save_message_to_history(chat_id, "assistant", response)

    except Exception as e:
        logger.error(f"[bold red]❌ Error processing {action_description} callback: {e}[/bold red]", exc_info=True)
        error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش درخواست شما پیش آمد.")
        try:
            bot.edit_message_text(
                error_message,
                chat_id=chat_id,
                message_id=message_id,
                parse_mode="MarkdownV2"
            )
        except Exception as edit_error:
            logger.error(f"Failed to edit message with error for {action_description}: {edit_error}")
            bot.send_message(chat_id, error_message, parse_mode="MarkdownV2") # Fallback
        save_message_to_history(chat_id, "assistant", "خطا در پردازش درخواست از منوی گزینه‌ها.")

def handle_coaching_with_ai(call):
    """Handles the 'Coaching with AI' button click."""
    logger.info(f"Processing 'coaching_with_ai' callback for chat {call.message.chat.id}")
    _process_action_callback(call, TODAY_COACHING_TIP_PROMPT, "کوچینگ با هوش مصنوعی")

def handle_instagram_story_idea(call):
    """Handles the 'Instagram Story Idea' button click."""
    logger.info(f"Processing 'instagram_story_idea' callback for chat {call.message.chat.id}")
    _process_action_callback(call, INSTA_IDEA_PROMPT, "ایده اینستا")

def handle_leaderboard(call):
    """Handles the 'Leaderboard' button click."""
    logger.info(f"Processing 'leaderboard' callback for chat {call.message.chat.id}")
    _process_action_callback(call, LEADER_BOARD_PROMPT, "رتبه‌بندی", is_leaderboard=True)
