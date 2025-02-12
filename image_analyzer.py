import logging
import g4f
from g4f.client import Client

client = Client()

def analyze_image(image, image_analyzer_prompt, user_text=None):
    """
    Analyze the given image with optional user text input.
    """
    prompt = ""
    if user_text:
        prompt += f"متن ورودی کاربر: {user_text}\n\n"
    prompt += image_analyzer_prompt
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            image=image
        )
        if hasattr(response.choices[0].message, 'content'):
            return response.choices[0].message.content
        elif isinstance(response.choices[0].message, dict):
            return response.choices[0].message.get('content', '')
        else:
            return str(response.choices[0].message)
    except Exception as e:
        logging.error(f"Error in image analysis: {e}")
        return "خطا در تحلیل تصویر. لطفاً دوباره تلاش کنید."
