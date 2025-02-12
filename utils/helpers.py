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

def refine_ai_response(response_md: str) -> str:
    """
    Refines the AI's markdown response for Telegram by replacing headings,
    list markers, and escaping special characters.
    """
    parts = response_md.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r'^####\s+(.*?)$', r'🔶 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^###\s+(.*?)$', r'⭐ \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^##\s+(.*?)$', r'🔷 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^#\s+(.*?)$', r'🟣 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*[-*]\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*\d+\.\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            temp = parts[i]
            placeholders = {
                'bold': (r'\*\*(.+?)\*\*', '‡B‡\\1‡B‡'),
                'italic': (r'\*(.+?)\*', '‡I‡\\1‡I‡'),
                'strike': (r'~~(.+?)~~', '‡S‡\\1‡S‡'),
                'code': (r'`(.+?)`', '‡C‡\\1‡C‡'),
                'link': (r'\[(.+?)\]\((.+?)\)', '‡L‡\\1‡U‡\\2‡L‡'),
            }
            for name, (pattern, repl) in placeholders.items():
                temp = re.sub(pattern, repl, temp)
            for char in special_chars:
                temp = temp.replace(char, f"\\{char}")
            restorations = {
                '‡B‡': '**',
                '‡I‡': '*',
                '‡S‡': '~~',
                '‡C‡': '`',
                '‡L‡': '[',
                '‡U‡': '](',
            }
            for placeholder, markdown in restorations.items():
                temp = temp.replace(placeholder, markdown)
            parts[i] = temp
        else:
            parts[i] = f'`{parts[i]}`'
    return ''.join(parts)

def escape_markdown_v2(text: str) -> str:
    """
    Escape special characters for Telegram's MarkdownV2 format.
    """
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    parts = text.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            for char in special_chars:
                parts[i] = parts[i].replace(char, f"\\{char}")
        else:
            parts[i] = f'`{parts[i]}`'
    return ''.join(parts)
