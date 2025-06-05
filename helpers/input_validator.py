import re
from ..constants.errors import MIN_INPUT_LENGTH

def clean_input(text: str) -> str:
    """Clean and normalize input text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters except Persian/Arabic
    text = re.sub(r'[^\u0600-\u06FF\s\w\d\.،,?؟!]', '', text)
    return text.strip()

def is_valid_input(text: str) -> bool:
    """Check if input meets minimum requirements."""
    if not text:
        return False
    cleaned = clean_input(text)
    return len(cleaned) >= MIN_INPUT_LENGTH

def get_persian_error(error_code: str) -> str:
    """Get localized error message."""
    from ..constants.errors import ERROR_MESSAGES
    return ERROR_MESSAGES.get(error_code, ERROR_MESSAGES['processing_failed'])
