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
import logging
from threading import Thread
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class HateSpeechDetector:
    def __init__(self):
        # Optimized for Render's memory constraints
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Reduced for memory efficiency
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_df=0.95,  # Ignore terms that appear in >95% of documents
            min_df=2      # Ignore terms that appear in <2 documents
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        self.is_trained = False
        self.training_in_progress = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        try:
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
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return ""
    
    def load_and_train_model(self, dataset_path):
        """Load dataset and train the model with error handling"""
        try:
            self.training_in_progress = True
            logger.info("Starting model training...")
            
            # Find CSV files in the dataset directory
            csv_files = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the dataset")
            
            # Load the first CSV file found
            logger.info(f"Loading dataset from: {csv_files[0]}")
            df = pd.read_csv(csv_files[0])
            
            # Log dataset info
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Identify text and label columns
            text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'tweet', 'comment', 'content'])]
            label_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['label', 'hate', 'class', 'target'])]
            
            if not text_columns or not label_columns:
                # Fallback: use first two columns
                text_col = df.columns[0]
                label_col = df.columns[1]
            else:
                text_col = text_columns[0]
                label_col = label_columns[0]
            
            logger.info(f"Using text column: {text_col}")
            logger.info(f"Using label column: {label_col}")
            
            # Sample data if too large (for memory constraints)
            if len(df) > 10000:
                logger.info("Sampling dataset for memory efficiency...")
                df = df.sample(n=10000, random_state=42)
            
            # Preprocess text data
            logger.info("Preprocessing text data...")
            df['processed_text'] = df[text_col].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['processed_text'].str.len() > 0]
            
            # Prepare features and labels
            X = df['processed_text']
            y = df[label_col]
            
            # Convert labels to binary
            if y.dtype == 'object':
                unique_labels = y.unique()
                logger.info(f"Unique labels: {unique_labels}")
                
                if len(unique_labels) == 2:
                    # Binary classification
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    y = y.map(label_map)
                else:
                    # Multi-class to binary conversion
                    hate_keywords = ['hate', 'offensive', 'abusive', '1', 'yes', 'true', 'toxic']
                    y = y.apply(lambda x: 1 if any(keyword in str(x).lower() for keyword in hate_keywords) else 0)
            
            # Ensure we have both classes
            if len(y.unique()) < 2:
                raise ValueError("Dataset must contain both hate speech and non-hate speech examples")
            
            logger.info(f"Class distribution: {y.value_counts().to_dict()}")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Vectorize the text
            logger.info("Vectorizing text data...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train the model
            logger.info("Training model...")
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info(f"Training completed successfully")
            
            self.is_trained = True
            
            # Save the model
            try:
                model_data = {
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'accuracy': accuracy
                }
                with open('hate_speech_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info("Model saved successfully")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
        finally:
            self.training_in_progress = False
    
    def predict(self, text):
        """Predict if text contains hate speech"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        if self.training_in_progress:
            return {"error": "Model training in progress, please wait"}
        
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {"error": "Text is empty after preprocessing"}
            
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
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}

# Initialize the detector
detector = HateSpeechDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "model_trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with the dataset"""
    if detector.training_in_progress:
        return jsonify({
            "status": "error",
            "message": "Training already in progress"
        })
    
    def train_async():
        try:
            logger.info("Starting async training...")
            # Download the dataset
            path = kagglehub.dataset_download("sharonibejih/nigerian-multilingual-hate-speech")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Train the model
            accuracy = detector.load_and_train_model(path)
            
            if accuracy is not None:
                logger.info(f"Training completed with accuracy: {accuracy}")
            else:
                logger.error("Training failed")
                
        except Exception as e:
            logger.error(f"Async training error: {str(e)}")
    
    # Start training in background
    thread = Thread(target=train_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": "Training started in background. Check /status for progress."
    })

@app.route('/predict', methods=['POST'])
def predict_hate_speech():
    """Predict hate speech for given text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
        result = detector.predict(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status')
def model_status():
    """Check if model is trained and ready"""
    return jsonify({
        "trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress,
        "status": "ready" if detector.is_trained else ("training" if detector.training_in_progress else "not trained")
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Auto-train model on startup (for production)
def auto_train_on_startup():
    """Automatically train model on startup if not already trained"""
    time.sleep(5)  # Wait for app to fully start
    if not detector.is_trained and not detector.training_in_progress:
        try:
            logger.info("Auto-training model on startup...")
            path = kagglehub.dataset_download("sharonibejih/nigerian-multilingual-hate-speech")
            detector.load_and_train_model(path)
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")

if __name__ == '__main__':
    # Try to load existing model
    try:
        with open('hate_speech_model.pkl', 'rb') as f:
            saved_model = pickle.load(f)
            detector.model = saved_model['model']
            detector.vectorizer = saved_model['vectorizer']
            detector.is_trained = True
            logger.info("Loaded existing model")
    except FileNotFoundError:
        logger.info("No existing model found.")
        # Start auto-training in background for production
        if os.environ.get('RENDER'):
            thread = Thread(target=auto_train_on_startup)
            thread.daemon = True
            thread.start()
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Always False in production
        threaded=True  # Enable threading for concurrent requests
    )