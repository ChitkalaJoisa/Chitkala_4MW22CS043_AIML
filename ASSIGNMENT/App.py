from flask import Flask, request, jsonify, render_template
from langdetect import detect, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Language code to full name mapping
language_mapping = {
    'en': 'English',
    'zh-cn':'Chinese',
    'ja': 'Japanese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'or': 'Odia',
    'pa': 'Punjabi',
    'as': 'Assamese',
    'mai': 'Maithili',
    'sa': 'Sanskrit',
    'doi': 'Dogri',
    'mni': 'Manipuri',
    'ks': 'Kashmiri',
    'sat': 'Santali',
    'sd': 'Sindhi',
    # Add more languages as needed
}

# Define the main route for the web app
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route that interacts with the frontend
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the input data from the frontend
    text = data['text']  # Extract the text field
    sentiment_scores = sid.polarity_scores(text)  # Get sentiment scores
    
    # Determine sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Calculate character and word count
    char_count = len(text)
    word_count = len(text.split())
    
    # Detect language
    try:
        language_code = detect(text)
        # Get full language name from mapping
        language = language_mapping.get(language_code, "Unknown Language")
    except LangDetectException:
        language = "Unknown Language"
    
    return jsonify({
        'sentiment': sentiment,
        'char_count': char_count,
        'word_count': word_count,
        'language': language
    })

if __name__ == '__main__':
    app.run(debug=True)
