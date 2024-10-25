from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import re
import os
import pickle
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model98.76.h5')

# Load TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the vectorizer fit on your training data
# For demonstration purposes, you'll need to use the actual vectorizer saved from training
# tfidf_vectorizer = joblib.load('path_to_tfidf_vectorizer.pkl')

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Implement the same preprocessing used for training
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    stopwords = set([
        'the', 'is', 'in', 'and', 'to', 'from', 'for', 'on', 'with', 'a', 'an',
        'of', 'at', 'by', 'this', 'that', 'it', 'as', 'are', 'was', 'were', 'be',
        'has', 'have', 'but', 'if', 'or', 'because', 'so', 'while', 'its', 'which',
        'about', 'who', 'whom', 'what', 'where', 'when', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
        'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
        'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    ])
    words = [word for word in words if word not in stopwords]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def predict(text):
    processed_text = preprocess(text)
    vectorized_text = tfidf_vectorizer.transform([processed_text]).toarray()
    vectorized_text = np.expand_dims(vectorized_text, axis=2)
    prediction = model.predict(vectorized_text)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    text = request.form['text']
    predictions = predict(text)
    results = {
        'toxic': predictions[0] > 0.5,
        'severe_toxic': predictions[1] > 0.5,
        'obscene': predictions[2] > 0.5,
        'threat': predictions[3] > 0.5,
        'insult': predictions[4] > 0.5,
        'identity_hate': predictions[5] > 0.5
    }
    return render_template('result.html', results=results, text=text)

if __name__ == '__main__':
    app.run(debug=True)
