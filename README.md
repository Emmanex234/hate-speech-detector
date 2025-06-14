# Nigerian Tweet Hate Speech Detector

A simple web application that detects hate speech in Nigerian tweets using machine learning. This project uses the "Nigerian Multilingual Hate Speech" dataset from Kaggle.

## Features

- **Multilingual Support**: Detects hate speech in Nigerian languages and English
- **Real-time Prediction**: Instant analysis of tweet content
- **Simple Web Interface**: Clean, responsive design
- **Model Training**: Automatic dataset download and model training
- **High Accuracy**: Uses TF-IDF vectorization with Logistic Regression

## Project Structure

```
hate-speech-detector/
│
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Internet connection (for dataset download)

### 2. Installation

1. **Clone or create the project directory**
   ```bash
   mkdir hate-speech-detector
   cd hate-speech-detector
   ```

2. **Create the files**
   - Copy the `app.py` code into a file named `app.py`
   - Create a `templates` folder and copy the HTML code into `templates/index.html`
   - Copy the requirements into `requirements.txt`

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 3. Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Train the model**
   - Click the "Train Model" button on the web interface
   - Wait for the dataset to download and model to train (this may take a few minutes)
   - The model will be saved for future use

4. **Start detecting hate speech**
   - Enter any tweet or text in the textarea
   - Click "Analyze Tweet" to get results
   - Try the example buttons for quick testing

## How It Works

### Backend (app.py)
- **Data Processing**: Downloads and preprocesses the Nigerian hate speech dataset
- **Model Training**: Uses TF-IDF vectorization and Logistic Regression
- **API Endpoints**: Provides REST API for training and prediction
- **Text Preprocessing**: Cleans tweets by removing URLs, mentions, and special characters

### Frontend (index.html)
- **Modern UI**: Clean, responsive design with animations
- **Real-time Status**: Shows model training status
- **Interactive Examples**: Quick test buttons with sample texts
- **Error Handling**: User-friendly error messages

### Model Details
- **Algorithm**: Logistic Regression with TF-IDF features
- **Features**: Up to 5000 TF-IDF features with 1-2 grams
- **Preprocessing**: URL removal, mention cleaning, punctuation removal
- **Performance**: Model accuracy displayed after training

## API Endpoints

### GET /
Returns the main web interface

### POST /train
Trains the hate speech detection model
- Downloads the dataset automatically
- Trains and evaluates the model
- Returns training status and accuracy

### POST /predict
Predicts hate speech for given text
- **Input**: JSON with `text` field
- **Output**: Prediction, confidence score, and probabilities

### GET /status
Returns current model status
- **Output**: Whether model is trained and ready

## Usage Examples

### Using the Web Interface
1. Open http://localhost:5000
2. Train the model (first time only)
3. Enter text to analyze
4. View results with confidence scores

### Using the API Directly

**Train Model:**
```bash
curl -X POST http://localhost:5000/train
```

**Predict Text:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your tweet text here"}'
```

**Check Status:**
```bash
curl http://localhost:5000/status
```

## Model Performance

The system uses a simple but effective approach:

- **TF-IDF Vectorization**: Converts text to numerical features
- **Logistic Regression**: Binary classification (hate/not hate)
- **Text Preprocessing**: Removes noise and normalizes text
- **Cross-validation**: Splits data for training and testing

Expected performance metrics:
- **Accuracy**: 75-85% (depending on dataset quality)
- **Training Time**: 2-5 minutes on average hardware
- **Prediction Time**: < 100ms per text

## Customization Options

### Modify the Model
Edit `app.py` to change model parameters:

```python
# Change TF-IDF parameters
self.vectorizer = TfidfVectorizer(
    max_features=10000,  # More features
    ngram_range=(1, 3),  # Include trigrams
    min_df=2,            # Minimum document frequency
)

# Try different models
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(n_estimators=100)
```

### Customize the Interface
Edit `templates/index.html` to:
- Change colors and styling
- Add more example texts
- Modify the layout
- Add additional features

### Add New Languages
The model automatically handles multilingual content from the dataset, but you can:
- Add language-specific preprocessing
- Include additional stop words
- Customize text cleaning for specific languages

## Troubleshooting

### Common Issues

**Model Training Fails:**
- Check internet connection
- Ensure Kaggle dataset is accessible
- Verify all dependencies are installed

**Predictions Are Inaccurate:**
- Retrain the model with more data
- Adjust preprocessing parameters
- Try different machine learning algorithms

**Web Interface Not Loading:**
- Check if Flask server is running
- Verify port 5000 is not blocked
- Check browser console for JavaScript errors

**Import Errors:**
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

### Performance Optimization

**For Better Accuracy:**
1. Increase TF-IDF features: `max_features=10000`
2. Use ensemble methods: Random Forest or Gradient Boosting
3. Add more preprocessing steps
4. Experiment with different n-gram ranges

**For Faster Predictions:**
1. Reduce TF-IDF features: `max_features=2000`
2. Use simpler models
3. Cache vectorizer transformations
4. Optimize text preprocessing

## Security Considerations

- Input validation is implemented for text length
- CORS is enabled for cross-origin requests
- Model files are saved locally (not in cloud)
- No user data is stored permanently

## Future Enhancements

Possible improvements:
- **Deep Learning**: Use BERT or similar transformers
- **Multi-class Classification**: Detect different types of hate speech
- **Real-time Monitoring**: Process live Twitter feeds
- **User Feedback**: Allow users to correct predictions
- **Language Detection**: Identify and handle specific Nigerian languages
- **Batch Processing**: Analyze multiple tweets at once

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the dataset license on Kaggle for any restrictions on the hate speech data.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your setup matches the requirements
3. Test with the provided examples
4. Check Python and package versions

## Acknowledgments

- Dataset: "Nigerian Multilingual Hate Speech" by sharonibejih on Kaggle
- Built with Flask, scikit-learn, and modern web technologies
- Inspired by the need for multilingual hate speech detection in Nigerian social media