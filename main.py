# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the dataset
def load_data():
    data = pd.read_csv('music_genre_data.csv')  # Your dataset path
    return data

# Preprocess the data
def preprocess_data(data):
    stop_words = set(stopwords.words('english'))

    # Clean and tokenize lyrics
    data['lyrics'] = data['lyrics'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    
    # Split data into features and labels
    X = data['lyrics']
    y = data['genre']

    # Convert text to numerical representation using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    return X, y, vectorizer

# Split data into train and test sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
# train_model.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from preprocess import load_data, preprocess_data, split_data

# Main function to execute the training
def main():
    data = load_data()
    X, y, vectorizer = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

if __name__ == "__main__":
    main()
# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    lyrics = data['lyrics']
    lyrics_vectorized = vectorizer.transform([lyrics])
    prediction = model.predict(lyrics_vectorized)
    return jsonify({'genre': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
