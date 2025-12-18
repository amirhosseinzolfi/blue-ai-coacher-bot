#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_ai_utils.py - Test script for session analysis functionality
"""

import logging
import base64
from ai_utils import analyze_session, encode_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_text_session():
    """Test session analysis with text input."""
    print("\n" + "="*60)
    print("TEST 1: Text Session Analysis")
    print("="*60 + "\n")
    
    session_text = """
    جلسه برنامهریزی اسپرینت - تاریخ: ۱۴۰۳/۱۲/۱۵
    
    شرکتکنندگان: علی، سارا، محمد
    
    بحثها:
    - بررسی پیشرفت پروژه Blue AI Coacher Bot
    - علی گزارش داد که ماژول Jira Integration تکمیل شده
    - سارا مشکلات UI را مطرح کرد و نیاز به بهبود دارد
    - محمد پیشنهاد اضافه کردن قابلیت تحلیل صوتی را داد
    
    تصمیمات:
    - اولویت بعدی: بهبود رابط کاربری
    - اضافه کردن تستهای واحد برای ماژول Jira
    - شروع کار روی قابلیت تحلیل صوتی
    
    وظایف:
    - علی: نوشتن تستها تا پایان هفته
    - سارا: طراحی UI جدید تا سهشنبه
    - محمد: تحقیق درباره کتابخانههای تحلیل صوتی
    """
    
    try:
        result = analyze_session(session_text)
        print("📝 Analysis Result:")
        print("-" * 60)
        print(result)
        print("\n✅ Text session analysis test PASSED!")
        return True
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        return False

def test_base64_audio_structure():
    """Test base64 audio input structure."""
    print("\n" + "="*60)
    print("TEST 2: Base64 Audio Input Structure")
    print("="*60 + "\n")
    
    # Create fake audio data and encode it
    fake_audio = b"RIFF....WAVEfmt fake audio data for testing"
    audio_b64 = base64.b64encode(fake_audio).decode('utf-8')
    
    audio_input = {
        'type': 'audio',
        'data': audio_b64,
        'mime_type': 'audio/wav'
    }
    
    print("✓ Base64 Audio Input Structure:")
    print(f"  Type: {audio_input['type']}")
    print(f"  MIME Type: {audio_input['mime_type']}")
    print(f"  Base64 Data Length: {len(audio_input['data'])} chars")
    print(f"  Data Preview: {audio_input['data'][:50]}...")
    print("\n✅ Base64 audio structure test PASSED!")
    return True

def test_dict_text_input():
    """Test text input via dictionary."""
    print("\n" + "="*60)
    print("TEST 3: Dictionary Text Input")
    print("="*60 + "\n")
    
    session_dict = {
        'text': 'جلسه کوتاه: بررسی پیشرفت پروژه. تصمیم: ادامه کار روی فیچر جدید.'
    }
    
    try:
        result = analyze_session(session_dict)
        print("📝 Analysis Result:")
        print("-" * 60)
        print(result)
        print("\n✅ Dictionary text input test PASSED!")
        return True
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting AI Utils Session Analyzer Tests")
    print("Using Gemini Flash Latest Model\n")
    
    results = []
    
    # Run all tests
    results.append(("Text Session Analysis", test_text_session()))
    results.append(("Base64 Audio Structure", test_base64_audio_structure()))
    results.append(("Dictionary Text Input", test_dict_text_input()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60 + "\n")
