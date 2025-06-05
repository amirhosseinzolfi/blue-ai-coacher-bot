"""
Blue Business Bot Prompt System
-------------------------------
This module provides access to all prompts used by the Blue Business Bot.
"""

from .prompts import (
    # Export uppercase constants
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
    SUMMARY_PROMPT_TEXT,
    get_all_prompts
)

# For backward compatibility, provide lowercase aliases
prompt_template_text = PROMPT_TEMPLATE_TEXT
daily_task_prompt = DAILY_TASK_PROMPT
summary_prompt = SUMMARY_PROMPT
daily_report_prompt = DAILY_REPORT_PROMPT
insta_idea_prompt = INSTA_IDEA_PROMPT
image_analyzer_prompt = IMAGE_ANALYZER_PROMPT
business_info_summary_prompt = BUSINESS_INFO_SUMMARY_PROMPT
welcome_message = WELCOME_MESSAGE
help_text = HELP_TEXT
user_report_prompt = USER_REPORT_PROMPT
summary_prompt_text = SUMMARY_PROMPT_TEXT

# New prompt helper functions
def get_system_instruction(kid_info, ai_tone, conversation_context, user_name):
    """Create system instruction incorporating business info and tone."""
    system_template = f"""You are Blue, a professional AI business coach.

Business Information: {kid_info}

AI Tone: {ai_tone}

User: {user_name}

Recent Conversation Context:
{conversation_context}

Remember to be concise, professional, and respond in Persian (Farsi).
"""
    return system_template

def get_summary_prompt(conversation):
    """Generate a summary prompt for conversation history."""
    return f"Summarize the following conversation briefly, focusing on key topics and business-related information:\n\n{conversation}"

def get_image_prompt(description):
    """Generate an image prompt based on description."""
    return f"Create a professional business-themed image based on: {description}"

def get_info_prompt(info_text):
    """Generate a prompt to analyze business information."""
    return f"Analyze and summarize the following business information:\n\n{info_text}"

def get_ai_tone_prompt(tone_text):
    """Generate a prompt to process AI tone settings."""
    return f"Summarize and optimize the following AI tone description:\n\n{tone_text}"

# Constants
DEFAULT_AI_TONE = "دوستانه"
KIDS_INFO = "اطلاعات پیش فرض"  # Note: We're using business info instead
LOADING_MESSAGE_TEXT = "در حال بارگیری..."
MARKDOWN_V2_PARSE_MODE = "MarkdownV2"
