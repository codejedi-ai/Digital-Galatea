"""Emotional State Agent - responsible for managing and updating emotional state"""
import os
import sys
import random
import logging
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

class EmotionalStateAgent:
    """Agent responsible for managing and updating emotional state"""
    
    def __init__(self, initial_state=None, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.emotional_state = initial_state or {"joy": 0.2, "sadness": 0.2, "anger": 0.2, "fear": 0.2, "curiosity": 0.2}
        self.learning_rate = 0.05
        self.quantum_random_available = False
        self.quantum_api_key = None
        self._initialize_quantum()
    
    def _initialize_quantum(self):
        """Initialize quantum randomness availability"""
        quantum_key = os.getenv("ANU_QUANTUM_API_KEY")
        if quantum_key:
            self.quantum_api_key = quantum_key
            self.quantum_random_available = True
            logging.info("[EmotionalStateAgent] ✓ Quantum randomness available")
        else:
            logging.warning("[EmotionalStateAgent] Quantum randomness unavailable")
    
    def get_quantum_random_float(self, min_val=0.0, max_val=1.0):
        """Get a quantum random float between min_val and max_val"""
        if not self.quantum_random_available:
            return random.uniform(min_val, max_val)
        
        try:
            quantum_config = self.config.get('quantum', {}) if self.config else {}
            url = quantum_config.get('api_endpoint', 'https://api.quantumnumbers.anu.edu.au')
            headers = {"x-api-key": self.quantum_api_key}
            params = {"length": 1, "type": "uint8"}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and 'data' in result and len(result['data']) > 0:
                    normalized = result['data'][0] / 255.0
                    return min_val + (max_val - min_val) * normalized
        except Exception as e:
            logging.warning(f"[EmotionalStateAgent] Quantum API failed: {e}")
        
        return random.uniform(min_val, max_val)
    
    def update_with_sentiment(self, sentiment_score):
        """Update emotional state based on sentiment"""
        # Enhanced Emotion Update (decay and normalization with quantum randomness)
        decay_factor = 0.9
        if self.quantum_random_available:
            quantum_decay_variation = self.get_quantum_random_float(0.85, 0.95)
            decay_factor = quantum_decay_variation
        
        for emotion in self.emotional_state:
            # Decay emotions (more realistic fading with quantum variation)
            self.emotional_state[emotion] *= decay_factor
            # Normalize
            self.emotional_state[emotion] = max(0.0, min(1.0, self.emotional_state[emotion]))

        # Apply sentiment with quantum-enhanced learning rate variation
        learning_rate = self.learning_rate
        if self.quantum_random_available:
            quantum_lr_variation = self.get_quantum_random_float(0.03, 0.07)
            learning_rate = quantum_lr_variation
        
        self.emotional_state["joy"] += sentiment_score * learning_rate
        self.emotional_state["sadness"] -= sentiment_score * learning_rate
        
        # Add quantum randomness to curiosity (making responses more unpredictable)
        if self.quantum_random_available:
            quantum_curiosity_boost = self.get_quantum_random_float(-0.05, 0.05)
            self.emotional_state["curiosity"] = max(0.0, min(1.0, 
                self.emotional_state["curiosity"] + quantum_curiosity_boost))

        # Re-normalize
        total_emotion = sum(self.emotional_state.values())
        for emotion in self.emotional_state:
            self.emotional_state[emotion] = self.emotional_state[emotion] / total_emotion if total_emotion > 0 else 0.2
        
        logging.info(f"[EmotionalStateAgent] Updated emotional state: {self.emotional_state}")
        return self.emotional_state
    
    def get_state(self):
        """Get current emotional state"""
        return self.emotional_state.copy()
    
    def is_ready(self):
        """Check if agent is ready"""
        return True  # Emotional state is always ready

