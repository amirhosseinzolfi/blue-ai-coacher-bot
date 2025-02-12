import re

def refine_ai_response(response_md: str) -> str:
    # Refines AI's markdown response for modern Telegram view.
    parts = response_md.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r'^####\s+(.*?)$', r'🔶 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^###\s+(.*?)$', r'⭐ \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^##\s+(.*?)$', r'🔷 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^#\s+(.*?)$', r'🟣 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*[-*]\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            parts[i] = re.sub(r'^(?:\s*\d+\.\s+)(.*?)$', r'🔹 \1', parts[i], flags=re.MULTILINE)
            # ...other placeholder processing...
        else:
            parts[i] = f'`{parts[i]}`'
    return ''.join(parts)

def escape_markdown_v2(text: str) -> str:
    # Escape special characters for Telegram MarkdownV2.
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    parts = text.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            for char in special_chars:
                parts[i] = parts[i].replace(char, f'\\{char}')
        else:
            parts[i] = f'`{parts[i]}`'
    return ''.join(parts)

def format_multimodal_input(input_val):
    # Formats a list of text and image_url blocks as a string.
    if isinstance(input_val, list):
        parts = []
        for block in input_val:
            if block.get("type") == "text":
                parts.append(block.get("data", ""))
            elif block.get("type") == "image_url":
                parts.append(f"[Image: {block.get('data', '')}]")
        return "\n".join(parts)
    return str(input_val)
