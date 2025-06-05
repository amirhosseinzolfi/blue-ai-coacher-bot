"""
message_handlers.py - Handles text, photo, and document message processing
Part of the Blue AI Coacher Bot system.
"""

import io
import json
import os
import datetime
import re
import requests
import threading  # Add this import
from data_extractors import extract_text_from_file, get_supported_file_types, validate_file_size, setup_data_extractors

from utils.helpers import escape_markdown_v2
from config import (
    TELEGRAM_BOT_TOKEN,
    business_info_update_pending
)
from db_manager import save_message_to_history, save_business_info
from langgraph_code import (
    run_agent,
    process_business_info,
    summarize_business_info,
    image_analyze_llm as langgraph_image_analyze_llm # Import the specific LLM instance
)
from image_agent import generate_image

# Logger will be injected from bot.py
logger = None
# Bot instance will be injected from bot.py
bot = None

# Define the lock at the module level
task_file_lock = threading.Lock()

# Add a new global dictionary to store pending files for batch processing
pending_business_files = {}
file_batch_lock = threading.Lock()

def setup_message_handlers(bot_instance, logger_instance):
    """Initialize this module with the bot instance and logger"""
    global bot, logger
    bot = bot_instance
    logger = logger_instance
    
    # Initialize data extractors
    setup_data_extractors(logger_instance)
    
    # Return a dictionary of message handler functions
    return {
        'text_photo': handle_message,
        'document': handle_document,
        'task_entry': handle_task_entry,
        'image_generation': process_image_generation
    }

def handle_message(message):
    """
    Main handler for text and photo messages
    """
    chat_type = message.chat.type
    chat_id = str(message.chat.id)
    sender_first_name = message.from_user.first_name or message.from_user.username

    # Check if user is requesting the menu
    if message.content_type == 'text' and any(keyword in message.text.lower() for keyword in 
        ["menu", "منو", "دکمه", "keyboard", "کیبورد", "button", "buttons", "دکمه‌ها"]):
        from command_handlers import get_main_menu_keyboard
        bot.send_message(chat_id, "منوی اصلی:", reply_markup=get_main_menu_keyboard())
        save_message_to_history(chat_id, "system", "منوی اصلی نمایش داده شد.")
        return

    # Define the menu button texts to ignore in this handler
    menu_buttons = [
        "➕ افزودن تسک", "📊 گزارش امروز",  # Changed "➕ افزودن کار امروز" to "➕ افزودن تسک"
        "🎨 ساخت تصویر", "⚙️ گزینه‌های بیشتر"
        # Old buttons removed: "➕ تسک", "✅ تسک امروز", "💡 کوچینگ", "📸 اینستا", "🏆 رتبه‌بندی"
        # Note: "💡 کوچینگ", "📸 اینستا", "🏆 رتبه‌بندی" are typically handled by callbacks from /options,
        # so they wouldn't appear as raw text from main menu.
        # "🎨 تصویر" is covered by "🎨 ساخت تصویر".
        # "📊 گزارش امروز" is kept.
    ]

    # If the message text is one of the menu buttons, let main_menu_handler deal with it
    if message.content_type == 'text' and message.text in menu_buttons:
        logger.debug(f"Ignoring message '{message.text}' in handle_message as it's handled by main_menu_handler.")
        return # Stop processing here

    logger.info("[bold blue]" + "-"*40 + "[/bold blue]")
    logger.info(f"[blue]💬 New message from {sender_first_name} in {chat_type} {chat_id}[/blue]")

    if business_info_update_pending.get(chat_id) and message.content_type == 'text':
        logger.info(f"Received text for business info update from chat {chat_id}")
        new_info_raw = message.text
        
        # Send a waiting message to the user while processing
        waiting_message = bot.reply_to(message, escape_markdown_v2("🔍 در حال تحلیل اطلاعات کسب و کار با هوش مصنوعی..."), parse_mode="MarkdownV2")
        
        # Check for existing business info
        from db_manager import get_business_info
        existing_info = get_business_info(chat_id)
        
        # Use the business LLM to summarize the business info
        try:
            # First process the raw text to clean it
            processed_info = process_business_info(new_info_raw, chat_id)
            
            # If there's existing info, create a structured prompt combining both
            if existing_info and existing_info.strip():
                combined_info = (
                    "# اطلاعات کسب و کار موجود\n"
                    f"{existing_info}\n\n"
                    "# اطلاعات کسب و کار جدید\n"
                    f"{processed_info}\n\n"
                    "# دستورالعمل\n"
                    "لطفاً اطلاعات فوق را ترکیب کنید تا یک توصیف جامع و بروزرسانی شده از کسب و کار ایجاد شود. "
                    "اطلاعات جدید را در اولویت قرار دهید، اما اطلاعات مفید قبلی را حفظ کنید. "
                    "پاسخ نهایی باید کامل‌ترین نمای کسب و کار را ارائه دهد."
                )
                # Then summarize it with the business LLM
                final_info = summarize_business_info(combined_info)
                logger.info(f"Combined business info summarized successfully for chat {chat_id}")
            else:
                # No existing info, just summarize the new info
                final_info = summarize_business_info(processed_info)
                logger.info(f"New business info summarized successfully for chat {chat_id}")
            
            # Save the summarized info
            save_business_info(chat_id, final_info, chat_type)
            business_info_update_pending.pop(chat_id, None)

            # Update the waiting message with success message
            response_text = f"✅ اطلاعات کسب و کار با موفقیت بروزرسانی شد."
            bot.edit_message_text(escape_markdown_v2(response_text), chat_id=chat_id, message_id=waiting_message.message_id, parse_mode="MarkdownV2")
            
            # Show the full processed business information to the user
            bot.send_message(
                chat_id,
                escape_markdown_v2(f"📋 خلاصه اطلاعات بیزینس ثبت شده:\n\n{final_info}"),
                parse_mode="MarkdownV2"
            )
            
            save_message_to_history(chat_id, "user", f"{sender_first_name}: [ارسال اطلاعات کسب و کار]")
            save_message_to_history(chat_id, "system", "اطلاعات کسب و کار بروزرسانی شد.")
            
        except Exception as e:
            logger.error(f"Error summarizing business info: {e}")
            processed_info = process_business_info(new_info_raw, chat_id)  # Fallback to just basic processing
            final_info = processed_info  # Fallback to processed but unsummarized text
            
            # Save the processed info as fallback
            save_business_info(chat_id, final_info, chat_type)
            business_info_update_pending.pop(chat_id, None)
            
            # Update the waiting message with a fallback success message
            response_text = f"✅ اطلاعات کسب و کار ذخیره شد (بدون تحلیل هوش مصنوعی)."
            bot.edit_message_text(escape_markdown_v2(response_text), chat_id=chat_id, message_id=waiting_message.message_id, parse_mode="MarkdownV2")
            
            # Show the full processed business information to the user
            bot.send_message(
                chat_id,
                escape_markdown_v2(f"📋 خلاصه اطلاعات بیزینس ثبت شده:\n\n{final_info}"),
                parse_mode="MarkdownV2"
            )
            
            save_message_to_history(chat_id, "user", f"{sender_first_name}: [ارسال اطلاعات کسب و کار]")
            save_message_to_history(chat_id, "system", "اطلاعات کسب و کار ذخیره شد (بدون تحلیل هوش مصنوعی).")
        
        return

    if message.content_type == 'photo':
        text_component = message.caption if message.caption else ""
        query_payload = []
        if text_component:
            query_payload.append({"type": "text", "text": text_component})
        query_payload.append({"type": "image_url", "image_url": {"url": f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{bot.get_file(message.photo[-1].file_id).file_path}"}})
        user_message_text = text_component if text_component else "تصویر دریافت شد."
        if message.reply_to_message and message.reply_to_message.text:
            replied_text = message.reply_to_message.text.strip()
            # Optimized format for multimodal with reply
            query_payload = [
                {"type": "text", "text": f"""
👤 user: {sender_first_name}
💬 MESSAGE: {text_component}
↩️ REPLYING TO: "{replied_text}"
📷 IMAGE: Attached below"""},
                {"type": "image_url", "image_url": {"url": f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{bot.get_file(message.photo[-1].file_id).file_path}"}}
            ]
            user_message_text = f'(Replying to: "{replied_text}")\n{user_message_text}'
    else:
        main_text = message.text.strip()
        if message.reply_to_message and message.reply_to_message.text:
            replied = message.reply_to_message.text.strip()
            # Optimized format for text with reply
            query_payload = f"""
👤 user: {sender_first_name}
💬 MESSAGE: {main_text}
↩️ REPLYING TO: "{replied}"

Please respond to the user's message, taking into account the context of the message they're replying to."""
            user_message_text = f'(Reply to "{replied}") {main_text}'
        else:
            # Optimized format for simple text message
            query_payload = f"""
👤 user: {sender_first_name}
💬 MESSAGE: {main_text}

Please respond to this message appropriately."""
            user_message_text = main_text

    logger.info(f"[bold cyan]💬 Message saved: {user_message_text[:50]}{'...' if len(user_message_text)>50 else ''}[/bold cyan]")
    save_message_to_history(chat_id, "user", f"{sender_first_name}: {user_message_text}")

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
        bot.edit_message_text(refined_response, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
        # save assistant response in background
        threading.Thread(
            target=save_message_to_history,
            args=(chat_id, "assistant", refined_response),
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"[bold red]❌ Error: {str(e)}[/bold red]")
        
        # Check if this was a photo message and try image analysis fallback
        if message.content_type == 'photo':
            try:
                logger.info("🖼️ Attempting image analysis fallback for photo message...")
                
                # Use the imported image_analyze_llm from langgraph_code
                # No need to redefine it here
                # from langchain_openai import ChatOpenAI
                # image_analyze_llm = ChatOpenAI(
                #     base_url="http://141.98.210.149:15403/v1", # This was the old one
                #     model_name="gpt-4o",
                #     temperature=0.5,
                #     api_key="1"
                # )
                
                # Get business info for context
                from db_manager import get_business_info
                business_info = get_business_info(chat_id)
                
                # Create comprehensive image analysis prompt
                from prompts.prompts import IMAGE_ANALYZER_PROMPT
                
                image_analyzer_system_prompt = f"""You are a specialized image analyzer with advanced visual understanding capabilities.
                Your task is to:
                1. Provide a detailed, comprehensive description of the image
                2. Answer the user's specific text query related to the image
                3. Provide contextual analysis based on the business information provided
                
                {IMAGE_ANALYZER_PROMPT}
                
                Business Context: {business_info}
                
                Please analyze the image thoroughly and respond to the user's request in Persian (Farsi).
                Be detailed in your description and helpful in your response.
                """
                
                # Prepare multimodal content for image analyzer
                text_component = message.caption if message.caption else "لطفا این تصویر را تحلیل کنید."
                multimodal_content = [
                    {"type": "text", "text": f"User ({sender_first_name}): {text_component}"},
                    {"type": "image_url", "image_url": {"url": f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{bot.get_file(message.photo[-1].file_id).file_path}"}}
                ]
                
                # Send to image analyzer
                from langchain_core.messages import SystemMessage, HumanMessage
                image_analysis_messages = [
                    SystemMessage(content=image_analyzer_system_prompt),
                    HumanMessage(content=multimodal_content)
                ]
                
                logger.info("Sending image to specialized analyzer...")
                # Use the imported langgraph_image_analyze_llm
                image_analyzer_response = langgraph_image_analyze_llm.invoke(image_analysis_messages)
                image_analysis_text = image_analyzer_response.content
                
                logger.info(f"✅ Image analyzer provided analysis: {image_analysis_text[:100]}...")
                
                # Now try the main agent with text-only input containing the analysis
                comprehensive_query = f"""کاربر ({sender_first_name}) تصویری همراه با این متن ارسال کرد: "{text_component}"

تحلیل کامل تصویر:
{image_analysis_text}

لطفاً بر اساس این تحلیل و درخواست کاربر، پاسخ مناسبی ارائه دهید."""
                
                logger.info("🔄 Retrying with image analysis as text input...")
                refined_response = run_agent(comprehensive_query, chat_id, placeholder_message.message_id, sender_first_name)
                
                bot.edit_message_text(refined_response, chat_id=chat_id,
                                    message_id=placeholder_message.message_id,
                                    parse_mode="MarkdownV2")
                
                # Save both the image analysis and final response to history
                threading.Thread(
                    target=save_message_to_history,
                    args=(chat_id, "system", f"تحلیل تصویر: {image_analysis_text}"),
                    daemon=True
                ).start()
                
                threading.Thread(
                    target=save_message_to_history,
                    args=(chat_id, "assistant", refined_response),
                    daemon=True
                ).start()
                
                logger.info("✅ Successfully processed image with fallback method")
                return
                
            except Exception as fallback_error:
                logger.error(f"❌ Image analysis fallback also failed: {fallback_error}", exc_info=True)
                error_message = escape_markdown_v2(f"""❌ متأسفم، مشکلی در تحلیل تصویر و پردازش درخواست شما پیش آمد.

لطفاً موارد زیر را بررسی کنید:
• تصویر واضح و قابل خواندن باشد
• اتصال اینترنت شما پایدار باشد
• دوباره تلاش کنید یا تصویر را به صورت جداگانه ارسال کنید

اگر مشکل ادامه داشت، می‌توانید متن درخواست خود را بدون تصویر ارسال کنید.""")
        else:
            error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش درخواست شما پیش آمد.")
        
        bot.edit_message_text(error_message, chat_id=chat_id,
                          message_id=placeholder_message.message_id,
                          parse_mode="MarkdownV2")
    logger.info("[bold blue]" + "-"*40 + "[/bold blue]")

def process_accumulated_files(chat_id: str, chat_type: str):
    """
    Process all accumulated files for business info update
    """
    global pending_business_files
    
    with file_batch_lock:
        if chat_id not in pending_business_files:
            return
        
        files_data = pending_business_files[chat_id]
        if not files_data['files']:
            return
        
        logger.info(f"Processing {len(files_data['files'])} accumulated files for chat {chat_id}")
        
        # Extract text from all files
        all_extracted_texts = []
        successfully_processed_files = []
        failed_files = []
        
        for file_info in files_data['files']:
            try:
                extracted_text = extract_text_from_file(
                    file_info['content'], 
                    file_info['name'], 
                    file_info['mime_type']
                )
                all_extracted_texts.append(f"=== فایل: {file_info['name']} ===\n{extracted_text}\n\n")
                successfully_processed_files.append(file_info['name'])
                logger.info(f"Successfully extracted {len(extracted_text)} characters from {file_info['name']}")
            except Exception as e:
                logger.error(f"Failed to extract text from {file_info['name']}: {e}")
                failed_files.append(f"{file_info['name']}: {str(e)}")
        
        # Combine all extracted texts
        if all_extracted_texts:
            combined_text = "\n".join(all_extracted_texts)
            combined_header = f"=== مجموعه اطلاعات کسب و کار از {len(successfully_processed_files)} فایل ===\n"
            combined_header += f"فایل‌های پردازش شده: {', '.join(successfully_processed_files)}\n"
            if failed_files:
                combined_header += f"فایل‌های ناموفق: {', '.join(failed_files)}\n"
            combined_header += "=" * 70 + "\n\n"
            
            new_info_raw = combined_header + combined_text
            
            # Update the waiting message
            waiting_message_id = files_data['waiting_message_id']
            bot.edit_message_text(
                escape_markdown_v2("🔍 در حال تحلیل تمام فایل‌ها با هوش مصنوعی..."), 
                chat_id=chat_id, 
                message_id=waiting_message_id, 
                parse_mode="MarkdownV2"
            )
            
            # Check for existing business info
            from db_manager import get_business_info
            existing_info = get_business_info(chat_id)
            
            # Process and summarize business info with the business LLM
            try:
                processed_info = process_business_info(new_info_raw, chat_id)
                
                # If there's existing info, create a structured prompt combining both
                if existing_info and existing_info.strip():
                    combined_info = (
                        "# اطلاعات کسب و کار موجود\n"
                        f"{existing_info}\n\n"
                        "# اطلاعات کسب و کار جدید از فایل‌ها\n"
                        f"{processed_info}\n\n"
                        "# دستورالعمل\n"
                        "لطفاً اطلاعات فوق را ترکیب کنید تا یک توصیف جامع و بروزرسانی شده از کسب و کار ایجاد شود. "
                        "اطلاعات جدید را در اولویت قرار دهید، اما اطلاعات مفید قبلی را حفظ کنید. "
                        "پاسخ نهایی باید کامل‌ترین نمای کسب و کار را ارائه دهد."
                    )
                    final_info = summarize_business_info(combined_info)
                    logger.info(f"Combined business info from {len(successfully_processed_files)} files summarized successfully for chat {chat_id}")
                else:
                    final_info = summarize_business_info(processed_info)
                    logger.info(f"Business info from {len(successfully_processed_files)} files summarized successfully for chat {chat_id}")
                
                # Save the summarized info
                save_business_info(chat_id, final_info, chat_type)
                business_info_update_pending.pop(chat_id, None)
                
                # Update the waiting message with success message
                if failed_files:
                    response_text = f"✅ اطلاعات کسب و کار از {len(successfully_processed_files)} فایل با موفقیت بروزرسانی شد.\n\n⚠️ {len(failed_files)} فایل پردازش نشد."
                else:
                    response_text = f"✅ اطلاعات کسب و کار از {len(successfully_processed_files)} فایل با موفقیت بروزرسانی شد."
                
                bot.edit_message_text(escape_markdown_v2(response_text), chat_id=chat_id, message_id=waiting_message_id, parse_mode="MarkdownV2")
                
                # Show the full processed business information to the user
                bot.send_message(
                    chat_id,
                    escape_markdown_v2(f"📋 خلاصه اطلاعات بیزینس ثبت شده:\n\n{final_info}"),
                    parse_mode="MarkdownV2"
                )
                
                save_message_to_history(chat_id, "user", f"[ارسال {len(successfully_processed_files)} فایل اطلاعات کسب و کار: {', '.join(successfully_processed_files)}]")
                save_message_to_history(chat_id, "system", f"اطلاعات کسب و کار از {len(successfully_processed_files)} فایل بروزرسانی شد.")
                
            except Exception as e:
                logger.error(f"Error summarizing business info from files: {e}")
                processed_info = process_business_info(new_info_raw, chat_id)  # Fallback to just basic processing
                
                # Save the processed info as fallback
                save_business_info(chat_id, processed_info, chat_type)
                business_info_update_pending.pop(chat_id, None)
                
                # Update the waiting message with a fallback success message
                response_text = f"✅ اطلاعات کسب و کار ذخیره شد (بدون تحلیل هوش مصنوعی)."
                bot.edit_message_text(escape_markdown_v2(response_text), chat_id=chat_id, message_id=waiting_message_id, parse_mode="MarkdownV2")
                
                # Show the full processed business information to the user
                bot.send_message(
                    chat_id,
                    escape_markdown_v2(f"📋 خلاصه اطلاعات بیزینس ثبت شده:\n\n{processed_info}"),
                    parse_mode="MarkdownV2"
                )
                
                save_message_to_history(chat_id, "user", f"[ارسال {len(successfully_processed_files)} فایل اطلاعات کسب و کار: {', '.join(successfully_processed_files)}]")
                save_message_to_history(chat_id, "system", f"اطلاعات کسب و کار از {len(successfully_processed_files)} فایل ذخیره شد (بدون تحلیل هوش مصنوعی).")
        
        else:
            # No files could be processed
            waiting_message_id = files_data['waiting_message_id']
            error_msg = "❌ هیچ فایلی قابل پردازش نبود."
            if failed_files:
                error_msg += f"\n\nخطاها:\n" + "\n".join([f"• {error}" for error in failed_files])
            
            bot.edit_message_text(
                escape_markdown_v2(error_msg), 
                chat_id=chat_id, 
                message_id=waiting_message_id, 
                parse_mode="MarkdownV2"
            )
        
        # Clear the pending files for this chat
        del pending_business_files[chat_id]

def schedule_file_processing(chat_id: str, chat_type: str, delay_seconds: int = 3):
    """
    Schedule file processing after a delay to allow for multiple file uploads
    """
    def delayed_process():
        import time
        time.sleep(delay_seconds)
        process_accumulated_files(chat_id, chat_type)
    
    # Cancel any existing timer for this chat
    if chat_id in pending_business_files and 'timer' in pending_business_files[chat_id]:
        pending_business_files[chat_id]['timer'].cancel()
    
    # Start new timer
    timer = threading.Timer(delay_seconds, delayed_process)
    timer.start()
    
    if chat_id in pending_business_files:
        pending_business_files[chat_id]['timer'] = timer
    
    logger.info(f"Scheduled file processing for chat {chat_id} in {delay_seconds} seconds")

def handle_document(message):
    """
    Handler for document messages (primarily for business info updates)
    """
    global pending_business_files
    
    chat_id = str(message.chat.id)
    chat_type = message.chat.type
    sender_first_name = message.from_user.first_name or message.from_user.username
    
    if business_info_update_pending.get(chat_id):
        logger.info(f"Received document for business info update from chat {chat_id}")
        
        # Validate file size (20MB limit)
        if not validate_file_size(message.document.file_size):
            bot.reply_to(
                message, 
                escape_markdown_v2("❌ حجم فایل بیش از حد مجاز است. حداکثر حجم مجاز: ۲۰ مگابایت"), 
                parse_mode="MarkdownV2"
            )
            return
        
        try:
            # Download file
            file_info = bot.get_file(message.document.file_id)
            file_bytes = bot.download_file(file_info.file_path)
            
            # Store file information for batch processing
            with file_batch_lock:
                if chat_id not in pending_business_files:
                    # Send initial waiting message
                    waiting_message = bot.reply_to(
                        message, 
                        escape_markdown_v2("📁 فایل دریافت شد. در انتظار فایل‌های بیشتر... (۳ ثانیه)"), 
                        parse_mode="MarkdownV2"
                    )
                    
                    pending_business_files[chat_id] = {
                        'files': [],
                        'waiting_message_id': waiting_message.message_id,
                        'chat_type': chat_type
                    }
                else:
                    # Update existing waiting message
                    current_count = len(pending_business_files[chat_id]['files']) + 1
                    bot.edit_message_text(
                        escape_markdown_v2(f"📁 {current_count} فایل دریافت شد. در انتظار فایل‌های بیشتر... (۳ ثانیه)"), 
                        chat_id=chat_id, 
                        message_id=pending_business_files[chat_id]['waiting_message_id'], 
                        parse_mode="MarkdownV2"
                    )
                
                # Add current file to the batch
                pending_business_files[chat_id]['files'].append({
                    'name': message.document.file_name,
                    'mime_type': message.document.mime_type,
                    'content': file_bytes,
                    'size': message.document.file_size
                })
                
                logger.info(f"Added file {message.document.file_name} to batch for chat {chat_id}. Total files: {len(pending_business_files[chat_id]['files'])}")
            
            # Schedule processing (this will reset the timer if called multiple times)
            schedule_file_processing(chat_id, chat_type, delay_seconds=3)
            
        except Exception as e:
            logger.error(f"Error handling business info document: {e}")
            error_message = f"❌ خطا در دریافت فایل: {str(e)}"
            bot.reply_to(message, escape_markdown_v2(error_message), parse_mode="MarkdownV2")
    else:
        logger.debug(f"Received document in chat {chat_id}, but not expecting business info.")
        # Handle other document types here if needed
        
        # General document handling for regular chat
        logger.info(f"Received document in chat {chat_id} for regular processing")
        
        # Validate file size (20MB limit)
        if not validate_file_size(message.document.file_size):
            bot.reply_to(
                message, 
                escape_markdown_v2("❌ حجم فایل بیش از حد مجاز است. حداکثر حجم مجاز: ۲۰ مگابایت"), 
                parse_mode="MarkdownV2"
            )
            return
            
        try:
            # Send a waiting message
            waiting_message = bot.reply_to(message, escape_markdown_v2("📄 در حال استخراج متن از فایل..."), parse_mode="MarkdownV2")
            
            # Download file
            file_info = bot.get_file(message.document.file_id)
            file_bytes = bot.download_file(file_info.file_path)
            file_name = message.document.file_name
            
            # Extract text from file
            try:
                extracted_text = extract_text_from_file(
                    file_bytes, 
                    file_name, 
                    message.document.mime_type
                )
                logger.info(f"Successfully extracted {len(extracted_text)} characters from {file_name}")
                
                # Format the message with file content
                file_content_formatted = f"## فایل آپلود شده توسط کاربر:\n\n{extracted_text}"
                
                # If file was in response to another message, include context
                context_text = ""
                if message.reply_to_message and message.reply_to_message.text:
                    replied_text = message.reply_to_message.text.strip()
                    context_text = f"""
👤 user: {sender_first_name}
💬 MESSAGE: فایل {file_name} ارسال شد
↩️ REPLYING TO: "{replied_text}"

## متن استخراج شده از فایل:
{file_content_formatted}

لطفا متن فایل را تحلیل کنید و با توجه به پیام کاربر پاسخ دهید."""
                else:
                    # Regular file with no reply context
                    context_text = f"""
👤 user: {sender_first_name}
💬 MESSAGE: فایل {file_name} ارسال شد

## متن استخراج شده از فایل:
{file_content_formatted}

لطفا محتوای فایل را تحلیل کنید و پاسخ مناسبی ارائه دهید."""
                
                # Save to chat history
                save_message_to_history(chat_id, "user", f"{sender_first_name}: [ارسال فایل: {file_name}]")
                
                # Delete waiting message and notify
                bot.delete_message(chat_id, waiting_message.message_id)
                
                # Send typing indication
                bot.send_chat_action(chat_id, 'typing')
                placeholder_message = bot.reply_to(message, escape_markdown_v2("🤔 در حال تحلیل محتوای فایل..."), parse_mode="MarkdownV2")
                
                # Process with AI
                try:
                    refined_response = run_agent(context_text, chat_id, placeholder_message.message_id, sender_first_name)
                    
                    # Update message with response
                    bot.edit_message_text(refined_response, chat_id=chat_id,
                                        message_id=placeholder_message.message_id,
                                        parse_mode="MarkdownV2")
                    
                    # Save assistant response
                    threading.Thread(
                        target=save_message_to_history,
                        args=(chat_id, "assistant", refined_response),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    logger.error(f"[bold red]❌ Error in AI processing of file: {str(e)}[/bold red]")
                    error_message = escape_markdown_v2("❌ متأسفم، مشکلی در تحلیل محتوای فایل پیش آمد.")
                    bot.edit_message_text(error_message, chat_id=chat_id,
                                      message_id=placeholder_message.message_id,
                                      parse_mode="MarkdownV2")
                
            except Exception as extraction_error:
                logger.error(f"Text extraction failed: {extraction_error}")
                
                # Show supported file types
                supported_types = get_supported_file_types()
                types_text = "، ".join(supported_types.values())
                error_msg = f"❌ خطا در استخراج متن از فایل.\n\nفرمت‌های پشتیبانی شده: {types_text}\n\nخطا: {str(extraction_error)}"
                bot.edit_message_text(
                    escape_markdown_v2(error_msg), 
                    chat_id=chat_id, 
                    message_id=waiting_message.message_id, 
                    parse_mode="MarkdownV2"
                )
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            bot.reply_to(message, escape_markdown_v2(f"❌ خطا در پردازش فایل: {str(e)}"), parse_mode="MarkdownV2")

def handle_task_entry(message):
    """
    Handle the user's task entry response after clicking "➕ افزودن تسک"
    Save the entry to users_task.json and database and respond with confirmation
    """
    chat_id = str(message.chat.id)
    user_name = message.from_user.first_name or message.from_user.username or "Unknown"
    task_text = message.text.strip()
    
    logger.info(f"Processing task entry from {user_name} in chat {chat_id}")
    
    if not task_text:
        bot.reply_to(message, "متن تسک خالی است. لطفا متن معتبری وارد کنید.")
        return
    
    # Generate timestamp and today's date
    today = datetime.date.today().isoformat()
    timestamp = datetime.datetime.now().isoformat()
    
    # Create storage directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(script_dir, "database")
    os.makedirs(db_dir, exist_ok=True)
    
    # Save to database first
    from db_manager import save_user_task
    db_save_success = save_user_task(chat_id, user_name, task_text, today, timestamp)
    
    # Also save to JSON file for backward compatibility
    json_save_success = False
    try:
        db_path = os.path.join(db_dir, "users_task.json")
        
        with task_file_lock:  # Use lock to prevent concurrent writes
            # Read existing data
            tasks_data = {}
            if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
                try:
                    with open(db_path, "r", encoding="utf-8") as f:
                        tasks_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in {db_path}. Creating new file.")
            
            # Initialize structure if needed
            if chat_id not in tasks_data:
                tasks_data[chat_id] = {}
            if today not in tasks_data[chat_id]:
                tasks_data[chat_id][today] = []
            
            # Ensure today's entries is a list - this fixes the 'append' error
            if not isinstance(tasks_data[chat_id][today], list):
                logger.warning(f"Converting non-list format to list for chat {chat_id}, today: {today}")
                
                # Convert dict format to list format if needed
                if isinstance(tasks_data[chat_id][today], dict):
                    old_data = tasks_data[chat_id][today]
                    new_data = []
                    
                    # Try to extract tasks from old format
                    for user, info in old_data.items():
                        if isinstance(info, dict) and "to_do" in info:
                            for task in info.get("to_do", []):
                                new_data.append({
                                    "entry": task,
                                    "timestamp": timestamp,  # Using current timestamp as fallback
                                    "user": user
                                })
                    
                    tasks_data[chat_id][today] = new_data
                else:
                    # If it's something else entirely, reset to empty list
                    tasks_data[chat_id][today] = []
            
            # Add new entry with timestamp
            new_entry = {
                "entry": task_text,
                "timestamp": timestamp,
                "user": user_name
            }
            
            # Now safe to append to today's entries
            tasks_data[chat_id][today].append(new_entry)
            
            # Write back to file with pretty formatting
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
                
            json_save_success = True
        
        logger.info(f"Successfully saved task entry for {user_name} in chat {chat_id} to JSON and DB")
        
        # Send confirmation to user
        confirmation_text = f"✅ تسک شما با موفقیت ثبت شد!\n\nتاریخ: {today}\nزمان: {timestamp.split('T')[1][:8]}"
        confirmation_message = bot.reply_to(message, escape_markdown_v2(confirmation_text), parse_mode="MarkdownV2")
        
        # Save to chat history
        save_message_to_history(chat_id, "user", task_text)
        save_message_to_history(chat_id, "system", confirmation_text)
        
        # Process the task message through the normal chat flow
        logger.info(f"Processing task message as regular chat input: '{task_text}'")
        
        # Send the task to AI for response
        bot.send_chat_action(chat_id, 'typing')
        placeholder_message = bot.send_message(chat_id, escape_markdown_v2("🤔 در حال بررسی تسک..."), parse_mode="MarkdownV2")
        
        try:
            # Optimized format for task entry
            query_payload = f"""### TASK ENTRY ###
👤 user: {user_name}
📝 TASK: {task_text}
📅 DATE: {today}
⏰ TIME: {timestamp.split('T')[1][:8]}

This is a new task entry. Please acknowledge and provide relevant guidance or feedback."""
            refined_response = run_agent(query_payload, chat_id, placeholder_message.message_id, user_name)
            
            bot.edit_message_text(refined_response, chat_id=chat_id,
                                message_id=placeholder_message.message_id,
                                parse_mode="MarkdownV2")
            
            # Save assistant response to history
            threading.Thread(
                target=save_message_to_history,
                args=(chat_id, "assistant", refined_response),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"[bold red]❌ Error processing task as chat: {str(e)}[/bold red]", exc_info=True)
            error_message = escape_markdown_v2("❌ متأسفم، مشکلی در پردازش تسک شما پیش آمد.")
            bot.edit_message_text(error_message, chat_id=chat_id,
                              message_id=placeholder_message.message_id,
                              parse_mode="MarkdownV2")
        
    except Exception as e:
        logger.error(f"Error saving task to JSON: {e}", exc_info=True)
        
        # If at least one storage method worked, still consider it a success
        if db_save_success:
            confirmation_text = "✅ تسک شما با موفقیت ثبت شد! (فقط در پایگاه داده اصلی)"
            bot.reply_to(message, escape_markdown_v2(confirmation_text), parse_mode="MarkdownV2")
        else:
            bot.reply_to(message, "❌ خطا در ذخیره تسک. لطفا دوباره تلاش کنید.")

def process_image_generation(message):
    """
    Process the user's image description and generate the image
    """
    chat_id = str(message.chat.id)
    user_prompt = message.text
    sender_first_name = message.from_user.first_name or message.from_user.username
    
    if not user_prompt or len(user_prompt.strip()) < 3:
        bot.reply_to(
            message,
            escape_markdown_v2("❌ لطفاً توضیح دقیق‌تری برای ساخت تصویر ارائه دهید."),
            parse_mode="MarkdownV2"
        )
        return
    
    logger.info(f"[bold cyan]🖼️ Processing image generation for prompt: '{user_prompt}' from {sender_first_name}[/bold cyan]")
    save_message_to_history(chat_id, "user", f"{sender_first_name}: {user_prompt}")
    placeholder_msg = bot.reply_to(
        message,
        escape_markdown_v2("🎨 در حال ساخت تصویر... لطفاً صبر کنید."),
        parse_mode="MarkdownV2"
    )
    
    try:
        model = "midjourney"
        if "dall-e" in user_prompt.lower() or "dalle" in user_prompt.lower():
            model = "dall-e-3"
        elif "flux" in user_prompt.lower():
            model = "flux"
        logger.info(f"[bold purple]Selected image model: {model}[/bold purple]")
        
        try:
            image_url = generate_image(user_prompt, model)
            logger.info(f"[bold green]✅ Image generated successfully with direct approach[/bold green]")
        except Exception as direct_error:
            logger.warning(f"Direct image generation failed: {direct_error}. Trying with agent...")
            full_prompt = f"Generate an image of {user_prompt}"
            from image_agent import agent as image_agent
            image_url = image_agent.run(full_prompt)
            logger.info(f"[bold green]✅ Image generated successfully with agent approach[/bold green]")
        
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_data = io.BytesIO(image_response.content)
        image_data.name = "generated_image.jpg"
        
        bot.delete_message(chat_id, placeholder_msg.message_id)
        bot.send_photo(chat_id, image_data)
        
        assistant_response = (
            f"[تصویر] تصویر با موفقیت براساس درخواست شما ایجاد شد.\n\n"
            f"درخواست: {user_prompt}"
        )
        save_message_to_history(chat_id, "assistant", assistant_response)
        logger.info(f"[bold green]✅ Image sent to {sender_first_name} in chat {chat_id}[/bold green]")
    except Exception as e:
        error_message = f"❌ متأسفانه در ایجاد تصویر خطایی رخ داد: {str(e)}"
        logger.error(f"[bold red]Error generating image: {str(e)}[/bold red]")
        bot.edit_message_text(
            escape_markdown_v2(error_message),
            chat_id=chat_id,
            message_id=placeholder_msg.message_id,
            parse_mode="MarkdownV2"
        )
        save_message_to_history(chat_id, "assistant", f"خطا در ایجاد تصویر: نتوانستم تصویر درخواستی شما را بسازم.")
