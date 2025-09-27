import functions_framework
from google.cloud import texttospeech
import base64

# Initialize the client once to be reused
tts_client = texttospeech.TextToSpeechClient()

@functions_framework.http
def get_kannada_speech(request):
    """HTTP Cloud Function to generate Kannada speech audio."""
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        request_json = request.get_json(silent=True)
        text_to_speak = request_json['text']

        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        
        # Configure the voice request for Kannada
        voice = texttospeech.VoiceSelectionParams(
            language_code="kn-IN", 
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Call the Google TTS API
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Encode the audio bytes in Base64 to send in a JSON response
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        
        return ({"audioContent": audio_base64}, 200, headers)

    except Exception as e:
        return ({"error": str(e)}, 500, headers)