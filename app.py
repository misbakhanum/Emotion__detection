from flask import Flask, request, jsonify, render_template
import os
import pickle
import librosa
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoder
model_path = 'modelForPrediction1.sav'
encoder_path = 'label_encoder.sav'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Function to extract features from audio
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with librosa.load(file_name) as (audio, sample_rate):
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(audio))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result

# Define route for file upload and prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get the uploaded file
        audio_file = request.files['file']
        file_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(file_path)

        # Extract features
        features = extract_feature(file_path)
        features = features.reshape(1, -1)

        # Predict emotion
        prediction_encoded = model.predict(features)
        prediction = label_encoder.inverse_transform(prediction_encoded)

        # Remove the audio file after processing
        os.remove(file_path)

        # Return the predicted emotion
        return jsonify({'emotion': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Create 'uploads' folder if it doesn't exist
    os.makedirs('uploads', exist_ok=True)

    # Run the app
    app.run(debug=True)
