"""Sentiment Agent - responsible for sentiment analysis (uses Azure with NLTK fallback)"""
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG
from agents.azure_agent import AzureTextAnalyticsAgent

class SentimentAgent:
    """Agent responsible for sentiment analysis (uses Azure with NLTK fallback)"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.azure_agent = AzureTextAnalyticsAgent(config=self.config)
        self.ready = False
        self._fallback_analyzer = None
        self._initialize()
    
    def _initialize(self):
        """Initialize sentiment analyzer"""
        # Try Azure first
        if self.azure_agent.is_ready():
            self.ready = True
            logging.info("[SentimentAgent] Using Azure Text Analytics")
            return
 
        # Azure not available – rely on fallback
        logging.info("[SentimentAgent] Using NLTK VADER fallback for sentiment analysis")
        self.ready = True  # Fallback available
 
    def analyze(self, text):
        """Analyze sentiment of text (tries Azure first, falls back to NLTK)"""
        # Try Azure first
        if self.azure_agent.is_ready():
            result = self.azure_agent.analyze(text)
            if result is not None:
                return result
 
        return self._fallback_analyze(text)
 
    def _fallback_analyze(self, text):
        """Fallback sentiment analysis using NLTK VADER"""
        try:
            if self._fallback_analyzer is None:
                from nltk.sentiment import SentimentIntensityAnalyzer
                self._fallback_analyzer = SentimentIntensityAnalyzer()

            scores = self._fallback_analyzer.polarity_scores(text)
            return scores['compound']  # Returns value between -1 and 1
        except Exception as e:
            logging.error(f"[SentimentAgent] Fallback failed: {e}")
            return 0.0
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.ready

