from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json
import logging
from gtts import gTTS
import io
import base64
import tempfile
import os
import google.generativeai as genai
import whisper
import numpy as np
from pydub import AudioSegment
from pydub.utils import which
from dotenv import load_dotenv
load_dotenv()
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('API-KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

whisper_model = whisper.load_model("medium")

def chatbot(request):
    return render(request, 'index.html')

@csrf_exempt
@require_http_methods(["POST"])
def ask_ai(request):
    try:
        body = json.loads(request.body.decode('utf-8'))
        user_message = body.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                'error': 'Message cannot be empty'
            }, status=400)
        
        logger.info(f"User message: {user_message}")
        
        ai_response = generate_gemini_response(user_message)
        
        logger.info(f"AI response: {ai_response}")
        
        return JsonResponse({
            'reply': ai_response,
            'success': True
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        logger.error(f"Error in ask_ai: {str(e)}")
        return JsonResponse({
            'error': 'An internal server error occurred'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def speech_to_text(request):
    try:
        if 'audio' not in request.FILES:
            return JsonResponse({
                'error': 'No audio file provided'
            }, status=400)
        
        audio_file = request.FILES['audio']
        language = request.POST.get('language', 'en')
        
        if language not in ['en', 'hi']:
            language = 'en'  
        
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
            for chunk in audio_file.chunks():
                temp_input.write(chunk)
            temp_input_path = temp_input.name
        
        try:
            # Convert audio to format compatible with Whisper
            audio_segment = AudioSegment.from_file(temp_input_path)
            
            # Convert to WAV format for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                audio_segment.export(temp_wav.name, format="wav")
                temp_wav_path = temp_wav.name
            
            # Transcribe using Whisper
            result = whisper_model.transcribe(
                temp_wav_path, 
                language=language,
                fp16=False 
            )
            
            transcript = result['text'].strip()
            detected_language = result.get('language', language)
            
            os.unlink(temp_input_path)
            os.unlink(temp_wav_path)
            
            return JsonResponse({
                'transcript': transcript,
                'detected_language': detected_language,
                'success': True
            })
            
        except Exception as e:
            # Clean up temp files on error
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in speech_to_text: {str(e)}")
        return JsonResponse({
            'error': 'Failed to process audio'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def text_to_speech(request):
    """Convert text to speech using gTTS"""
    import json
    import base64
    import os
    import tempfile
    import logging
    from django.http import JsonResponse
    from gtts import gTTS

    logger = logging.getLogger(__name__)

    try:
        body = json.loads(request.body.decode('utf-8'))
        text = body.get('text', '').strip()
        language = body.get('language', 'en')

        if not text:
            return JsonResponse({'error': 'Text cannot be empty'}, status=400)

        gtts_language_map = {'en': 'en', 'hi': 'hi'}

        if language not in gtts_language_map:
            return JsonResponse({'error': 'Unsupported language'}, status=400)

        gtts_lang = gtts_language_map[language]
        tts = gTTS(text=text, lang=gtts_lang, slow=False)

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            tts.save(temp_file.name)

        with open(temp_file.name, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        os.unlink(temp_file.name)

        return JsonResponse({'audio_data': audio_base64, 'success': True})

    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        return JsonResponse({'error': 'Failed to generate speech'}, status=500)


def generate_gemini_response(message):
    """
    Generate AI responses using Google's Gemini API
    """
    try:
        # Create a specilized prompt for constellation and astronomy topics
        prompt = f"""
        You are an enthusiastic and friendly astronomy assistant who loves teaching people about constellations and the night sky

        You help beginners understand stars constellations and space in a clear and exciting way

        When someone asks a question give a helpful accurate and engaging answer focused on astronomy especially constellations

        Do not use any punctuation marks emojis or special characters because your response will be converted to speech using a voice engine

        Speak clearly and naturally as if you are talking to someone who is curious about space

        If the question is not related to astronomy gently steer the conversation back to stars constellations or space exploration in a positive and helpful way

        User question {message}
"""


        # Generate response using Gemini
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating Gemini response: {str(e)}")
        return """🌟 I apologize, but I'm having trouble accessing my astronomical database right now. 

Please try asking your question again in a moment. I'm here to help you explore the wonders of the night sky, including constellations, stars, planets, and cosmic phenomena!

In the meantime, did you know that on any clear night, you can see about 2,000 to 3,000 stars with the naked eye? ✨"""
