#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ai_utils.py - AI Utilities for Session Analysis
Handles multimodal session analysis (voice/text) using Gemini Flash.
"""

import logging
import base64
from typing import Union, Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

def encode_audio(audio_path: str) -> str:
    """Encode audio file to base64."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

class SessionAnalyzer:
    """Analyzes sessions from multimodal input (voice or text)."""
    
    def __init__(self, api_key: str = GOOGLE_API_KEY):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
            temperature=0.3,
            api_key=api_key
        )
        logger.info("SessionAnalyzer initialized with gemini-2.0-flash-exp")
    
    def analyze_session(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Analyze a session from text or voice input.
        
        Args:
            input_data: Either a text string or dict:
                       For audio file: {'type': 'audio_file', 'path': 'path/to/audio.wav'}
                       For audio base64: {'type': 'audio', 'data': audio_b64, 'mime_type': 'audio/wav'}
                       For text: Just pass the string directly
        
        Returns:
            Session summary as string
        """
        prompt =""" 
تو یک دستیار هوشمند خلاصه‌ساز جلسات هستی. ورودی تو متن پیاده‌سازی‌شده‌ی یک فایل صوتی جلسه است. بر اساس این متن، یک گزارش کامل، کاربردی و جذاب تولید کن.

**سبک و لحن:**
- به هیچ وجه از فرمت جدول و markdown table  استفاده نکن
_ برایی جذاب تر شدن متن از ایموچی های مرتبت برای هر عنوان ور بخش های دیگه استفاده کن (زیاد استفاده نکن)
* فارسی روان، صمیمی و محترم (نه اداریِ خشک، نه شوخی‌دار).
* فقط اطلاعات واقعی گفته‌شده در جلسه را بنویس؛ هیچ چیز جدید، حدسی یا تخیلی اضافه نکن.
* اگر چیزی مشخص نیست، بنویس «در جلسه مشخص نشد».
* در هر بخش ۱–۲ ایموجی مرتبط برای جذابیت استفاده کن.

**خروجی را حتماً با Markdown و ساختار زیر بنویس:**

1. `# عنوان جلسه`
   یک عنوان کوتاه و جذاب که موضوع اصلی جلسه را بگوید.

2. `## خلاصه جلسه`
   یک پاراگراف ۳–۵ جمله‌ای که کل فضای جلسه، هدف و نتیجه کلی را توضیح دهد.

3. `## موضوعات اصلی جلسه`
   بولت‌پوینت از مهم‌ترین محورهای بحث.

4. `## تصمیم‌های نهایی`
   لیست تصمیم‌های قطعی (اگر تصمیمی نبود، بنویس: «در این جلسه تصمیم نهایی مشخص نشد.»).

5. `## وظایف و مسئولیت‌ها`
   برای هر کار:
   `نام مسئول و عنوان وظیفه و شرح کار و ددلاین به صورت لیست (اگر گفته شده)`

6. `## نکات مهم جلسه`
   ریسک‌ها، تذکرها، عدد و رقم‌های مهم، محدودیت‌ها و نکات حساس.

7. `## قدم‌های بعدی`
   لیست اقدام‌های بعدی مورد توافق و در صورت وجود، پیشنهادهایی که مطرح شده‌اند (با برچسب «پیشنهاد»).

**نکات تکمیلی:**

* اسم‌ها، تاریخ‌ها و اعداد را دقیق و شفاف بنویس.
* اگر بخشی از متن/صدا نامفهوم است، با عبارت «(بخش نامشخص در صدا/متن)» مشخص کن و چیزی جای آن اختراع نکن.
"""

        try:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            
            if isinstance(input_data, dict):
                if input_data.get('type') == 'audio_file':
                    # Load and encode audio file
                    audio_b64 = encode_audio(input_data['path'])
                    mime_type = input_data.get('mime_type', 'audio/wav')
                    content.append({
                        "type": "media",
                        "mime_type": mime_type,
                        "data": audio_b64
                    })
                elif input_data.get('type') == 'audio':
                    # Use provided base64 audio - Gemini format
                    mime_type = input_data.get('mime_type', 'audio/wav')
                    content.append({
                        "type": "media",
                        "mime_type": mime_type,
                        "data": input_data['data']
                    })
                else:
                    # Text in dict
                    text = input_data.get('text', '')
                    content[0]['text'] = f"{prompt}\n\nمتن جلسه:\n{text}"
            else:
                # Plain text string
                content[0]['text'] = f"{prompt}\n\nمتن جلسه:\n{input_data}"
            
            message = HumanMessage(content=content)
            response = self.llm.invoke([message])
            logger.info("Session analysis completed successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"Error analyzing session: {e}")
            raise

# Global instance
session_analyzer = SessionAnalyzer()

def analyze_session(input_data: Union[str, Dict[str, Any]]) -> str:
    """Convenience function to analyze a session."""
    return session_analyzer.analyze_session(input_data)

__all__ = ['SessionAnalyzer', 'session_analyzer', 'analyze_session', 'encode_audio']
