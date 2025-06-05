import re
import warnings
# Suppress Pydantic serializer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

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

def strip_thinking_tags(text):
    """
    Remove <think>...</think> blocks from the beginning of AI responses.
    
    Args:
        text (str): The AI response text that may contain thinking tags
        
    Returns:
        str: The cleaned text with thinking tags removed
    """
    import re
    
    # Handle None or non-string input
    if not text or not isinstance(text, str):
        return "" if text is None else text
    
    # Pattern to match <think>...</think> at the beginning of text
    # Using re.DOTALL to make '.' match newlines as well
    pattern = r'^<think>.*?</think>\s*'
    
    # Replace the pattern with empty string
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Also check for thinking tags that might not be at the very beginning (with some whitespace)
    if cleaned_text.lstrip().startswith("<think>"):
        pattern = r'\s*<think>.*?</think>\s*'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return cleaned_text.strip()

def escape_markdown_v2(text: str) -> str:
    """
    Escape special characters for Telegram's MarkdownV2 format.
    Characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
    Must be escaped with a preceding '\'.
    """
    if not isinstance(text, str):
        return ""
    
    # Characters that MUST be escaped:
    _escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    # Temporarily replace existing escape sequences to protect them
    protected_text = text
    for char_to_escape in _escape_chars:
        protected_text = protected_text.replace(f'\\{char_to_escape}', f'‡‡ESCAPED{ord(char_to_escape)}‡‡')

    # Now escape the remaining special characters
    escaped_text = protected_text
    for char_to_escape in _escape_chars:
        escaped_text = escaped_text.replace(char_to_escape, f'\\{char_to_escape}')
        
    # Restore the protected escape sequences
    final_text = escaped_text
    for char_to_escape in _escape_chars:
        final_text = final_text.replace(f'‡‡ESCAPED{ord(char_to_escape)}‡‡', f'\\{char_to_escape}')
        
    return final_text

def refine_ai_response(response) -> str:
    """
    Process AI response for clean, structured display in Telegram.
    Converts markdown to emoji-based formatting and handles special characters.
    """
    if not response:
        return ""
    
    # Handle response objects or extract content from LLM response types
    if isinstance(response, dict) and "messages" in response:
        for message in response["messages"]:
            if hasattr(message, "content") and hasattr(message, "type") and message.type == "assistant":
                response = message.content
                break
    
    # Extract string content from AIMessage or similar objects
    if hasattr(response, "content"):
        response = response.content
    
    # Convert response to string if it's not already
    response = str(response)
    
    # Strip thinking tags if present
    response = strip_thinking_tags(response)
    
    # Process code blocks separately
    parts = response.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:  # This is regular text, not a code block
            # Replace headers with emoji prefixes
            parts[i] = re.sub(r'^####\s+(.*?)$', r'🔵 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^###\s+(.*?)$', r'⭐ \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^##\s+(.*?)$', r'🔷 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^#\s+(.*?)$', r'🟣 \1', parts[i], flags=re.MULTILINE)

            # Replace bullet points and numbered lists with emoji
            parts[i] = re.sub(r'^(?:\s*[-*]\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*\d+\.\s+)(.*?)$', r'▫️ \1', parts[i], flags=re.MULTILINE)
            
            # Replace text formatting with visual alternatives
            parts[i] = re.sub(r'\*\*(.+?)\*\*', r'\1', parts[i])  # Bold
            parts[i] = re.sub(r'\*(.+?)\*', r'𝘐: \1', parts[i])      # Italic
            parts[i] = re.sub(r'__(.+?)__', r'𝑼: \1', parts[i])      # Underline
            parts[i] = re.sub(r'~~(.+?)~~', r'𝕊: \1', parts[i])      # Strikethrough
            
            # Replace quotes with a visual indicator
            parts[i] = re.sub(r'^>\s*(.*?)$', r'💬 \1', parts[i], flags=re.MULTILINE)
            
            # Replace links with clear text
            parts[i] = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1 (\2)', parts[i])
    
        else:  # This is a code block
            # Add code block indicator and format properly
            if ":" in parts[i]:  # Has language specification
                lang, code = parts[i].split(":", 1)
                parts[i] = f"💻 کد {lang.strip()}:\n{code}"
            else:
                parts[i] = f"💻 کد:\n{parts[i]}"
    
    # Join parts back together
    formatted_text = ''.join(parts)
    
    # Add visual structure improvements
    formatted_text = formatted_text.replace('\n\n\n', '\n\n')  # Remove excessive newlines
    formatted_text = re.sub(r'\n{4,}', '\n\n', formatted_text)  # Limit maximum consecutive newlines
    
    # Escape special characters for Telegram MarkdownV2
    formatted_text = escape_markdown_v2(formatted_text)
    
    # Right align every line with RTL marker (using Unicode RLM)
    lines = formatted_text.splitlines()
    formatted_text = "\n".join(["\u200F" + line for line in lines])
    
    return formatted_text
