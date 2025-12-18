# Session Assistant Feature

## Overview
Added a new **Session Assistant** feature to the Blue AI Coacher Bot that allows users to analyze meeting sessions using voice recordings or text input.

## Implementation

### 1. Core Module: `ai_utils.py`
- **SessionAnalyzer class**: Uses Gemini Flash (gemini-2.0-flash-exp) for multimodal analysis
- **analyze_session()**: Convenience function for session analysis
- **encode_audio()**: Helper to encode audio files to base64
- Supports three input types:
  - Plain text string
  - Audio file path: `{'type': 'audio_file', 'path': 'audio.wav'}`
  - Base64 audio: `{'type': 'audio', 'data': base64_string, 'mime_type': 'audio/wav'}`

### 2. Bot Integration

#### `command_handlers.py`
- Added "🎙️ دستیار جلسات" button to `/options` menu inline keyboard

#### `callback_handlers.py`
- **handle_session_assistant()**: Handles button click, prompts user for voice/text input
- Registered in callback handlers dictionary

#### `message_handlers.py`
- **process_session_analysis()**: Processes voice or text input
  - Downloads and encodes voice messages to base64
  - Sends to SessionAnalyzer for analysis
  - Returns structured summary to user

## User Flow

1. User clicks `/options` command
2. Clicks "🎙️ دستیار جلسات" button
3. Bot prompts: "Send voice recording or text of the session"
4. User sends:
   - 🎤 Voice message (recorded session)
   - 📝 Text message (typed session notes)
5. Bot analyzes and returns:
   - Main topics discussed
   - Decisions made
   - Tasks and responsibilities assigned
   - Key points and next actions

## Analysis Output Format
The AI provides structured Markdown output in Persian including:
- موضوعات اصلی بحث شده (Main topics)
- تصمیمات گرفته شده (Decisions made)
- وظایف و مسئولیتهای تعیین شده (Tasks assigned)
- نکات کلیدی و اقدامات بعدی (Key points and next actions)

## Technical Details

### Model
- **Primary**: gemini-2.0-flash-exp
- **Temperature**: 0.3 (for consistent analysis)
- **Multimodal**: Supports both audio and text input

### Audio Processing
- Telegram voice messages downloaded as bytes
- Encoded to base64
- Sent with proper MIME type (audio/ogg or audio/mpeg)
- Follows LangChain multimodal message format

### Error Handling
- API quota exceeded: Graceful error message
- Invalid input: User-friendly Persian error messages
- File size limits: Handled by existing Telegram constraints

## Testing

### Test Files
1. **test_ai_utils.py**: Unit tests for SessionAnalyzer
2. **test_session_integration.py**: Integration tests for bot feature

### Test Results
✅ All imports successful
✅ SessionAnalyzer initialization
✅ Text analysis working
✅ Multimodal structure validated

## Files Modified
- `ai_utils.py` (NEW)
- `command_handlers.py` (MODIFIED)
- `callback_handlers.py` (MODIFIED)
- `message_handlers.py` (MODIFIED)

## Usage Example

```python
from ai_utils import analyze_session

# Text input
result = analyze_session("جلسه برنامه‌ریزی: بحث درباره پروژه جدید...")

# Audio input (base64)
result = analyze_session({
    'type': 'audio',
    'data': audio_base64_string,
    'mime_type': 'audio/ogg'
})
```

## Future Enhancements
- Support for multiple audio files
- Session comparison and tracking
- Export session summaries to PDF
- Integration with task management system
