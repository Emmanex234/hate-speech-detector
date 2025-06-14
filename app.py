import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import pickle
import kagglehub

app = Flask(__name__)
CORS(app)

class HateSpeechDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_and_train_model(self, dataset_path):
        """Load dataset and train the model"""
        try:
            # Try to find CSV files in the dataset directory
            csv_files = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the dataset")
            
            # Load the first CSV file found
            df = pd.read_csv(csv_files[0])
            
            # Print dataset info
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
            
            # Identify text and label columns (adapt based on actual dataset structure)
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower() or 'comment' in col.lower()]
            label_columns = [col for col in df.columns if 'label' in col.lower() or 'hate' in col.lower() or 'class' in col.lower()]
            
            if not text_columns or not label_columns:
                # Fallback: use first column as text, second as label
                text_col = df.columns[0]
                label_col = df.columns[1]
            else:
                text_col = text_columns[0]
                label_col = label_columns[0]
            
            print(f"Using text column: {text_col}")
            print(f"Using label column: {label_col}")
            
            # Preprocess text data
            df['processed_text'] = df[text_col].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['processed_text'].str.len() > 0]
            
            # Prepare features and labels
            X = df['processed_text']
            y = df[label_col]
            
            # Convert labels to binary if needed (hate speech = 1, not hate speech = 0)
            if y.dtype == 'object':
                unique_labels = y.unique()
                print(f"Unique labels: {unique_labels}")
                # Map labels to binary (adjust based on your dataset)
                if len(unique_labels) == 2:
                    y = y.map({unique_labels[0]: 0, unique_labels[1]: 1})
                else:
                    # For multi-class, map hate speech indicators to 1, others to 0
                    hate_keywords = ['hate', 'offensive', 'abusive', '1', 'yes', 'true']
                    y = y.apply(lambda x: 1 if any(keyword in str(x).lower() for keyword in hate_keywords) else 0)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Vectorize the text
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train the model
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            self.is_trained = True
            
            # Save the model and vectorizer
            with open('hate_speech_model.pkl', 'wb') as f:
                pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
            
            return accuracy
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None
    
    def predict(self, text):
        """Predict if text contains hate speech"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(text)
            
            # Vectorize the text
            text_vec = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_vec)[0]
            probability = self.model.predict_proba(text_vec)[0]
            
            return {
                "text": text,
                "prediction": "Hate Speech" if prediction == 1 else "Not Hate Speech",
                "confidence": float(max(probability)),
                "hate_probability": float(probability[1]) if len(probability) > 1 else 0.0
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize the detector
detector = HateSpeechDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with the dataset"""
    try:
        # Download the dataset
        print("Downloading dataset...")
        path = kagglehub.dataset_download("sharonibejih/nigerian-multilingual-hate-speech")
        print(f"Dataset downloaded to: {path}")
        
        # Train the model
        accuracy = detector.load_and_train_model(path)
        
        if accuracy is not None:
            return jsonify({
                "status": "success",
                "message": "Model trained successfully",
                "accuracy": accuracy
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to train model"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Training error: {str(e)}"
        })

@app.route('/predict', methods=['POST'])
def predict_hate_speech():
    """Predict hate speech for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"})
        
        result = detector.predict(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"})

@app.route('/status')
def model_status():
    """Check if model is trained and ready"""
    return jsonify({
        "trained": detector.is_trained,
        "status": "ready" if detector.is_trained else "not trained"
    })

if __name__ == '__main__':
    # Try to load existing model
    try:
        with open('hate_speech_model.pkl', 'rb') as f:
            saved_model = pickle.load(f)
            detector.model = saved_model['model']
            detector.vectorizer = saved_model['vectorizer']
            detector.is_trained = True
            print("Loaded existing model")
    except FileNotFoundError:
        print("No existing model found. Please train the model first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)