"""Azure Text Analytics Agent - responsible for Azure Text Analytics sentiment analysis"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

class AzureTextAnalyticsAgent:
    """Agent responsible for Azure Text Analytics sentiment analysis"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.azure_available = False
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Azure Text Analytics client"""
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            
            # Load environment variables so Azure credentials from .env are available
            load_dotenv()

            key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
            endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
            
            if key and endpoint:
                try:
                    credential = AzureKeyCredential(key)
                    self.client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
                    self.azure_available = True
                    logging.info("[AzureTextAnalyticsAgent] ✓ Initialized and ready")
                except Exception as e:
                    logging.warning(f"[AzureTextAnalyticsAgent] Failed to create client: {e}")
                    self.azure_available = False
            else:
                logging.warning("[AzureTextAnalyticsAgent] ✗ Azure credentials not found")
                self.azure_available = False
        except ImportError:
            logging.warning("[AzureTextAnalyticsAgent] ✗ Azure SDK not installed")
            self.azure_available = False
    
    def analyze(self, text):
        """Analyze sentiment using Azure Text Analytics"""
        if not self.azure_available or not self.client:
            return None
        
        try:
            result = self.client.analyze_sentiment(documents=[text])[0]
            if result.sentiment == 'positive':
                return result.confidence_scores.positive
            elif result.sentiment == 'negative':
                return -result.confidence_scores.negative
            else:
                return 0.0
        except Exception as e:
            logging.error(f"[AzureTextAnalyticsAgent] Error: {e}")
            return None
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.azure_available

