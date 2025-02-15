from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SpellCheck:
    def __init__(self):
        try:
            self.model_name = "facebook/wav2vec2-base-960h"
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
                        
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def load_audio(self, file_path: str):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise

    def convert_speech_to_text(self, file_path: str) -> str:
        try:
            waveform, sample_rate = self.load_audio(file_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            input_values = self.processor(
                waveform.squeeze(0),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0].lower()

            return transcription.strip()

        except Exception as e:
            logger.error(f"Error in speech to text conversion: {str(e)}")
            raise

    def analyze_pronunciation(self, audio_file: str, reference_text: str):
        try:
            transcribed_text = self.convert_speech_to_text(audio_file)
            
            def normalize_text(text: str):
                return re.sub(r'[^a-zA-Z ]', '', text.lower()).split()
            
            transcribed_words = normalize_text(transcribed_text)
            reference_words = normalize_text(reference_text)
            
            incorrect_words = [word for word in transcribed_words if word not in reference_words]
            
            total_reference_words = len(reference_words)
            correct_words = len([word for word in transcribed_words if word in reference_words])
            accuracy = correct_words / total_reference_words if total_reference_words > 0 else 0
            
            return {
                'transcribed_text': transcribed_text,
                'incorrect_words': incorrect_words,
                'accuracy': accuracy,
                'correct_words': correct_words,
                'total_words': total_reference_words
            }

        except Exception as e:
            logger.error(f"Error in pronunciation analysis: {str(e)}")
            raise

evaluator = SpellCheck()

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        reference_text = request.form.get('reference_text', '')

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        results = evaluator.analyze_pronunciation(file_path, reference_text)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
