"""Sentiment Agent - responsible for sentiment analysis (uses Azure, Hugging Face, or NLTK fallback)"""
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG
from agents.azure_agent import AzureTextAnalyticsAgent

# Import transformers with error handling
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    logging.warning("Transformers library not available. Using fallback sentiment analysis.")
    transformers_available = False

class SentimentAgent:
    """Agent responsible for sentiment analysis (uses Azure, Hugging Face, or NLTK fallback)"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.azure_agent = AzureTextAnalyticsAgent(config=self.config)
        self.sentiment_analyzer = None
        self.ready = False
        self._initialize()
    
    def _initialize(self):
        """Initialize sentiment analyzer"""
        # Try Azure first
        if self.azure_agent.is_ready():
            self.ready = True
            logging.info("[SentimentAgent] Using Azure Text Analytics")
            return
        
        # Fallback to Hugging Face
        sentiment_model = self.config.get('sentiment', {}).get('primary_model', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english') if self.config else 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
        
        if transformers_available:
            try:
                logging.info("[SentimentAgent] Initializing Hugging Face sentiment analyzer...")
                self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
                self.ready = True
                logging.info("[SentimentAgent] ✓ Initialized successfully")
            except Exception as e:
                logging.warning(f"[SentimentAgent] Hugging Face model failed: {e}, using fallback")
                self.sentiment_analyzer = None
                self.ready = True  # Fallback available
        else:
            self.ready = True  # Fallback available
    
    def analyze(self, text):
        """Analyze sentiment of text (tries Azure, then Hugging Face, then NLTK)"""
        # Try Azure first
        if self.azure_agent.is_ready():
            result = self.azure_agent.analyze(text)
            if result is not None:
                return result
        
        # Fallback to Hugging Face
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    return score
                elif 'negative' in label:
                    return -score
                else:
                    return 0.0
            except Exception as e:
                logging.error(f"[SentimentAgent] Error: {e}")
                return self._fallback_analyze(text)
        else:
            return self._fallback_analyze(text)
    
    def _fallback_analyze(self, text):
        """Fallback sentiment analysis using NLTK VADER"""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            return scores['compound']  # Returns value between -1 and 1
        except Exception as e:
            logging.error(f"[SentimentAgent] Fallback failed: {e}")
            return 0.0
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.ready

