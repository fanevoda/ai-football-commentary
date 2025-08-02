import os
import uuid
from elevenlabs import save
from elevenlabs.client import ElevenLabs

# Same API key as used in generate_commentary_audio
API_KEY = "sk_7545a47ab7e1bab1ef52a96e37b26ef0193ce8e4519d2a03"
MODEL_ID = "eleven_multilingual_v2"

client = ElevenLabs(api_key=API_KEY)

def synthesize_sample_line(voice_id, text="This is a voice preview."):
    output_path = f"temp_previews/preview_{uuid.uuid4().hex}.mp3"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=MODEL_ID
    )

    save(audio, output_path)
    return output_path


