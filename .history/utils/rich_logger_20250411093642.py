import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table
from rich import print as rprint
import json
import datetime
import inspect
import os
import traceback
from functools import wraps

# Create console instance for rich output
console = Console()

# Custom log levels for more specific logging
PROCESS_START = 25  # Between INFO and WARNING
PROCESS_END = 26
PROCESS_RESULT = 27
API_REQUEST = 28
API_RESPONSE = 29

# Register custom log levels with names
logging.addLevelName(PROCESS_START, "PROCESS_START")
logging.addLevelName(PROCESS_END, "PROCESS_END")
logging.addLevelName(PROCESS_RESULT, "RESULT")
logging.addLevelName(API_REQUEST, "API_REQUEST")
logging.addLevelName(API_RESPONSE, "API_RESPONSE")

# Add methods to Logger class for our custom levels
def process_start(self, message, *args, **kwargs):
    self.log(PROCESS_START, f"-- START -- {message}")

def process_end(self, message, *args, **kwargs):
    self.log(PROCESS_END, f"-- END -- {message}")

def process_result(self, message, *args, **kwargs):
    self.log(PROCESS_RESULT, message, *args, **kwargs)

def api_request(self, message, *args, **kwargs):
    self.log(API_REQUEST, message, *args, **kwargs)

def api_response(self, message, *args, **kwargs):
    self.log(API_RESPONSE, message, *args, **kwargs)

logging.Logger.process_start = process_start
logging.Logger.process_end = process_end
logging.Logger.process_result = process_result
logging.Logger.api_request = api_request
logging.Logger.api_response = api_response

def setup_logger(level=logging.INFO):
    """Set up and configure the rich logger."""
    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    # Format: [time] [level] [thread] [module:line] message
    FORMAT = "%(message)s"
    
    # Configure rich handler with our custom format
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        enable_link_path=True
    )
    
    # Set up basic configuration
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[rich_handler]
    )
    
    # Set specific levels for noisy libraries
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("telebot").setLevel(logging.WARNING)
    
    logger = logging.getLogger("blue_business")
    logger.info("Logger initialized. Level: %s", logging.getLevelName(level))
    
    return logger

def display_content(title, content, content_type="text"):
    """Display content in a structured, colorful panel."""
    if content_type == "json" and isinstance(content, (dict, list)):
        try:
            if isinstance(content, str):
                content = json.loads(content)
            syntax = Syntax(json.dumps(content, indent=2, ensure_ascii=False), "json", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title=f"[bold blue]{title}[/bold blue]", border_style="blue"))
        except Exception as e:
            console.print(f"[bold red]Error displaying JSON: {e}[/bold red]")
    elif content_type == "python":
        syntax = Syntax(str(content), "python", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title=f"[bold green]{title}[/bold green]", border_style="green"))
    elif content_type == "markdown":
        console.print(Panel(content, title=f"[bold magenta]{title}[/bold magenta]", border_style="magenta"))
    else:
        max_len = 2000
        if isinstance(content, str) and len(content) > max_len:
            content = content[:max_len] + "...[truncated]"
        console.print(Panel(str(content), title=f"[bold yellow]{title}[/bold yellow]", border_style="yellow"))

def log_function(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module = inspect.getmodule(func).__name__
            
            # Extract chat_id from args if present (common in Telegram handlers)
            chat_id = None
            if args and hasattr(args[0], 'chat'):
                chat_id = args[0].chat.id
            elif 'message' in kwargs and hasattr(kwargs['message'], 'chat'):
                chat_id = kwargs['message'].chat.id
                
            chat_info = f" [Chat: {chat_id}]" if chat_id else ""
            
            logger.process_start(f"[bold cyan]➡️ Starting {func_name}{chat_info}[/bold cyan]")
            start_time = datetime.datetime.now()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.process_end(f"[bold green]✅ Completed {func_name}{chat_info} in {duration:.2f}s[/bold green]")
                return result
            except Exception as e:
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.error(f"[bold red]❌ Error in {func_name}{chat_info} after {duration:.2f}s: {str(e)}[/bold red]")
                logger.debug(traceback.format_exc())
                raise
                
        return wrapper
    return decorator

def log_telegram_message(logger, message, direction):
    """Log Telegram messages with user info and content."""
    chat_id = message.chat.id
    user = message.from_user.first_name or message.from_user.username or str(message.from_user.id)
    
    if direction == "received":
        icon = "📩"
        color = "cyan"
    else:  # sent
        icon = "📤"
        color = "green"
        
    content_type = message.content_type
    
    if content_type == "text":
        content = message.text
        type_desc = "text"
    elif content_type == "photo":
        content = message.caption or "[No caption]"
        type_desc = "photo"
    elif content_type in ['document', 'video', 'audio']:
        content = message.caption or f"[{content_type} file]"
        type_desc = content_type
    else:
        content = f"[{content_type}]"
        type_desc = content_type
        
    # Truncate long messages for display
    if len(content) > 100:
        display_content_str = content[:100] + "..."
    else:
        display_content_str = content
        
    logger.info(f"[bold {color}]{icon} {direction.capitalize()} {type_desc} from {user} [Chat: {chat_id}]: {display_content_str}[/bold {color}]")
    
    # For debugging, log complete content at debug level
    if len(content) > 100:
        logger.debug(f"Complete message content: {content}")

def log_api_interaction(logger, endpoint, request_data, response_data, status_code=None):
    """Log API interactions with request and response data."""
    table = Table(title=f"API Interaction: {endpoint}")
    
    table.add_column("Component", style="cyan")
    table.add_column("Data", style="green")
    
    table.add_row("Request", str(request_data)[:200] + ("..." if len(str(request_data)) > 200 else ""))
    
    if status_code:
        status_style = "green" if 200 <= status_code < 300 else "red"
        table.add_row("Status Code", f"[{status_style}]{status_code}[/{status_style}]")
    
    table.add_row("Response", str(response_data)[:200] + ("..." if len(str(response_data)) > 200 else ""))
    
    console.print(table)
    
    logger.debug(f"Full request data: {request_data}")
    logger.debug(f"Full response data: {response_data}")

def log_summarization(logger, original_text, summarized_text, topic=None):
    """Log summarization results with before/after comparison."""
    title = f"📝 Summarization Result" + (f" - {topic}" if topic else "")
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Original Text", style="yellow", width=40)
    table.add_column("Summarized Text", style="green", width=40)
    
    orig_preview = (original_text[:500] + "...") if len(original_text) > 500 else original_text
    summ_preview = summarized_text
    
    table.add_row(orig_preview, summ_preview)
    console.print(table)
    
    compression_rate = (len(summarized_text) / len(original_text) * 100) if original_text else 0
    color = "green" if compression_rate < 50 else ("yellow" if compression_rate < 80 else "red")
    
    logger.process_result(f"[bold {color}]✨ Text summarized:[/bold {color}] [yellow]{len(original_text)}[/yellow] → [green]{len(summarized_text)}[/green] chars ([{color}]{compression_rate:.1f}%[/{color}] of original)")

# Replace complex table logging with minimal structured messages:
def log_llm_request(logger, chat_id, session_id=None, ai_tone=None, prompt=None, history_count=None, user_context_available=False):
    log_msg = f"LLM Request | Chat: {chat_id}"
    if session_id:
        log_msg += f", Session: {session_id}"
    if ai_tone:
        log_msg += f", Tone: {ai_tone}"
    if history_count is not None:
        log_msg += f", History: {history_count}"
    logger.info(log_msg)

def log_ai_interaction(logger, prompt, response, model_name=None, duration=None):
    logger.info("AI Interaction | Model: %s | Prompt: %.50s | Response: %.50s", model_name or "N/A", prompt, response)

# New function to log user and business data in a structured short table
def log_user_business_data(logger, chat_id, business_info=None, user_report=None):
    """
    Log a short summary table for business info and user report.
    """
    table = Table(title=f"[bold cyan]User & Business Data: Chat {chat_id}[/bold cyan]", show_header=True, header_style="bold blue")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Summary", style="green")
    
    if business_info:
        short_biz = business_info if len(business_info) < 100 else business_info[:97] + "..."
        table.add_row("Business Info", short_biz)
    else:
        table.add_row("Business Info", "None")
    
    if user_report:
        short_report = user_report if len(user_report) < 100 else user_report[:97] + "..."
        table.add_row("User Report", short_report)
    else:
        table.add_row("User Report", "None")
    
    console.print(table)
    logger.info(f"Displayed data for chat {chat_id}: Business Info: {bool(business_info)} | User Report: {bool(user_report)}")

def log_agent_execution(logger, chat_id, thread_id=None, username=None, query=None, is_multimodal=False):
    """
    Log minimal agent execution details in a structured format.
    
    The log includes: Chat ID, Thread ID, Username, and a truncated query.
    """
    summary = f"Query: {'Multimodal' if is_multimodal else (query[:50] + ('...' if len(query) > 50 else ''))}"
    table = Table(title="[bold magenta]Agent Execution Details[/bold magenta]", show_header=True, header_style="bold blue")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Chat ID", str(chat_id))
    if thread_id:
        table.add_row("Thread ID", str(thread_id))
    if username:
        table.add_row("Username", username)
    table.add_row("Query", summary)
    
    console.print(table)
    logger.info(f"[bold yellow]Running LangGraph workflow for thread {thread_id or 'unknown'}[/bold yellow]")

# New function to log comprehensive interaction details
def log_comprehensive_interaction(logger, chat_id, session_id, system_instruction, user_raw_prompt, user_report, conversation_summary, ai_tone=None):
    """
    Logs a comprehensive summary of key LLM interaction details in one rich table.
    """
    table = Table(title="[bold magenta]Comprehensive Interaction Log[/bold magenta]", show_header=True, header_style="bold blue")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Chat ID", str(chat_id))
    table.add_row("Session ID", str(session_id) if session_id else "N/A")
    if ai_tone:
        table.add_row("AI Tone", str(ai_tone))
    table.add_row("System Instruction", system_instruction if len(system_instruction) < 80 else system_instruction[:77] + "...")
    table.add_row("User Raw Prompt", user_raw_prompt if len(user_raw_prompt) < 80 else user_raw_prompt[:77] + "...")
    table.add_row("User Report", (user_report if user_report and len(user_report) < 80 
                    else (user_report[:77] + "...") if user_report else "None"))
    table.add_row("Conversation Summary", (conversation_summary if conversation_summary and len(conversation_summary) < 80 
                    else (conversation_summary[:77] + "...") if conversation_summary else "N/A"))
    
    console.print(table)
    logger.debug("Logged comprehensive interaction information.")
