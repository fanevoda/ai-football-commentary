import shutil

from flask import Flask, request, send_file, jsonify, render_template
import os
import uuid
from werkzeug.utils import secure_filename
from video_processor.video_pipeline import generate_commentated_video
from generate_audio.generate_audio import synthesize_sample_line


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    params = request.form.to_dict()  # Optional form parameters (like language)

    filename = secure_filename(video.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)

    # ✅ Ensure unique output filename to prevent conflicts
    unique_id = uuid.uuid4().hex
    output_filename = f"processed_{unique_id}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    video.save(input_path)

    try:
        generate_commentated_video(input_path, output_path, params)
        video_url = f"/static/processed/{output_filename}"
        shutil.move(output_path, os.path.join("static", "processed", output_filename))

        return render_template("index.html", video_url=video_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/preview-voice', methods=['POST'])
def preview_voice():
    data = request.get_json()
    voice_name = data.get("voice")
    preview_text = data.get("text", "This is a voice preview.")

    if not voice_name:
        return {"error": "Missing 'voice' parameter"}, 400

    # Voice name to ID mapping
    VOICE_MAP = {
        "bogdan": "5asM3ZxsegvXfXI5vqKQ",
        "andrei": "ANRS3e9rxJEXUpOhaPDb",
        "corina": "RjgBjNgGkuZd49zyCxIq",
        "eva": "RgXx32WYOGrd7gFNifSf",
        "martin": "Wl3O9lmFSMgGFTTwuS6f",
        "adam": "EXAVITQu4vr4xnSDxMaL",
        "rachel": "ErXwobaYiN019PkySvjV"
    }

    voice_id = VOICE_MAP.get(voice_name.lower())
    if not voice_id:
        return {"error": f"Voice '{voice_name}' not recognized"}, 400

    try:
        preview_path = synthesize_sample_line(voice_id, preview_text)
        return send_file(preview_path, mimetype="audio/mpeg")
    except Exception as e:
        return {"error": str(e)}, 500



if __name__ == '__main__':
    app.run(debug=True)
