from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SpeechEvaluator:
    def __init__(self):
        try:
            self.model_name = "openai/whisper-large-v3-turbo"
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            logger.info(f"Loaded model: {self.model_name}")
                        
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

    def convert_speech_to_text(self, file_path: str, language: str) -> str:
        try:
            waveform, sample_rate = self.load_audio(file_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            input_features = self.processor(
                waveform.squeeze(0),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()

            return transcription.strip()

        except Exception as e:
            logger.error(f"Error in speech to text conversion: {str(e)}")
            raise

    def analyze_pronunciation(self, audio_file: str, reference_text: str, language: str):
        try:
            transcribed_text = self.convert_speech_to_text(audio_file, language)
            
            def normalize_text(text: str):
                return re.sub(r'[^a-zA-ZÀ-ỹ ]', '', text.lower()).split()
            
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

evaluator = SpeechEvaluator()

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        reference_text = request.form.get('reference_text', '')
        language = request.form.get('language', 'english')

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        results = evaluator.analyze_pronunciation(file_path, reference_text, language)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
