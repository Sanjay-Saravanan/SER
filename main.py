from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import torch
import whisper
from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

whisper_model = whisper.load_model("base").to(device)
emotion_analyzer = pipeline("text-classification", model="michellejieli/emotion_text_classifier",
                            device=0 if torch.cuda.is_available() else -1)

ALLOWED_EXTENSIONS = {'mp3', 'm4a', 'mp4', 'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_audio_to_wav(file_path):
    try:
        base, ext = os.path.splitext(file_path)
        if ext.lower() == '.wav':
            return file_path
        
        wav_file = f"{base}.wav"
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_file, format="wav")

        return wav_file
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        return None

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Speech Emotion Recognition</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background-color: #0B0C10;
                color: #C5C6C7;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                text-align: center;
            }
            .container {
                background: #1F2833;
                padding: 50px;
                border-radius: 12px;
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5);
                width: 100%;
                max-width: 700px;
            }
            h1 {
                font-size: 4em;
                color: #66FCF1;
                margin-bottom: 20px;
            }
            p {
                font-size: 1.2em;
                line-height: 1.6;
                color: #C5C6C7;
            }
            input[type="file"] {
                display: block;
                margin: 20px auto;
                padding: 12px;
                font-size: 1.1em;
                font-weight: bold;
                color: #1F2833;
                background: #66FCF1;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }
            button {
                background-color: #66FCF1;
                color: #1F2833;
                font-size: 1.2em;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            button:hover {
                background-color: #45A29E;
            }
            #loading {
                display: none;
                margin-top: 20px;
                font-size: 1.5em;
                color: #66FCF1;
            }
            .spinner {
                border: 5px solid #C5C6C7;
                border-top: 5px solid #66FCF1;
                border-radius: 50%;
                +
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #result {
                margin-top: 30px;
                padding: 20px;
                background: #0B0C10;
                border: 2px solid #66FCF1;
                border-radius: 10px;
                text-align: left;
                max-height: 400px;
                overflow-y: auto;
            }
        </style>
        <script>
            async function uploadFile(event) {
                event.preventDefault();
                const formData = new FormData();
                formData.append('file', document.querySelector('#fileInput').files[0]);

                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                document.getElementById('loading').style.display = 'none';
                const resultDiv = document.querySelector('#result');
                resultDiv.innerHTML = '';

                result.forEach((segment, index) => {
                    const p = document.createElement('p');
                    p.innerHTML = `<strong>Segment ${index + 1}:</strong><br>Text: ${segment.text}<br>Emotions: ${JSON.stringify(segment.emotions)}`;
                    resultDiv.appendChild(p);
                });

                resultDiv.scrollIntoView({ behavior: 'smooth' });
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Speech Emotion Recognition</h1>
            <p>Upload an audio file, and we'll analyze the emotions behind the words.</p>
            <form id="uploadForm" onsubmit="uploadFile(event)">
                <input type="file" id="fileInput" name="file" accept=".mp3, .m4a, .mp4, .wav" required>
                <button type="submit">Upload and Analyze</button>
            </form>
            <div id="loading">
                <div class="spinner"></div>
                Analyzing audio, please wait...
            </div>
            <div id="result"></div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    wav_file_path = convert_audio_to_wav(file_path)
    if not wav_file_path:
        return jsonify({"error": "Failed to convert audio"}), 500

    result = whisper_model.transcribe(wav_file_path, verbose=False)

    segments = []
    for seg in result['segments']:
        text = seg['text']
        emotions = emotion_analyzer(text)
        segments.append({"text": text, "emotions": emotions})

    return jsonify(segments)

if __name__ == '__main__':
    app.run(debug=True)