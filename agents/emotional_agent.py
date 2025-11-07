"""Emotional State Agent - responsible for managing and updating emotional state"""
import os
import sys
import json
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
        
        # Get JSON file path from config
        emotions_config = self.config.get('emotions', {}) if self.config else {}
        self.json_path = emotions_config.get('json_path', './emotions.json')
        
        # Load emotional state from JSON file if it exists, otherwise use initial_state or defaults
        self.emotional_state = self._load_from_json() or initial_state or {"joy": 0.2, "sadness": 0.2, "anger": 0.2, "fear": 0.2, "curiosity": 0.2}
        
        # Slower learning rate for more gradual emotion changes
        self.learning_rate = 0.03
        self.quantum_random_available = False
        self.quantum_api_key = None
        self._initialize_quantum()
        
        # Save initial state to JSON
        self._save_to_json()
    
    def _load_from_json(self):
        """Load emotional state from JSON file"""
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # Validate state structure
                    required_emotions = ["joy", "sadness", "anger", "fear", "curiosity"]
                    if all(emotion in state for emotion in required_emotions):
                        logging.info(f"[EmotionalStateAgent] Loaded emotional state from {self.json_path}")
                        return state
                    else:
                        logging.warning(f"[EmotionalStateAgent] Invalid state structure in {self.json_path}, using defaults")
            else:
                logging.info(f"[EmotionalStateAgent] No existing emotional state file found, starting fresh")
        except Exception as e:
            logging.warning(f"[EmotionalStateAgent] Error loading emotional state from JSON: {e}")
        return None
    
    def _save_to_json(self):
        """Save emotional state to JSON file"""
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.emotional_state, f, indent=2, ensure_ascii=False)
            logging.debug(f"[EmotionalStateAgent] Saved emotional state to {self.json_path}")
        except Exception as e:
            logging.error(f"[EmotionalStateAgent] Error saving emotional state to JSON: {e}")
    
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
        # Slower decay factor to prevent emotions from crashing to minimum too fast
        # Changed from 0.9 to 0.97 (only 3% decay per interaction instead of 10%)
        decay_factor = 0.97
        if self.quantum_random_available:
            quantum_decay_variation = self.get_quantum_random_float(0.95, 0.99)
            decay_factor = quantum_decay_variation
        
        for emotion in self.emotional_state:
            # Decay emotions more slowly (preserves emotional state longer)
            self.emotional_state[emotion] *= decay_factor
            # Clamp to valid range
            self.emotional_state[emotion] = max(0.0, min(1.0, self.emotional_state[emotion]))

        # Apply sentiment with slower learning rate for gradual changes
        learning_rate = self.learning_rate
        if self.quantum_random_available:
            quantum_lr_variation = self.get_quantum_random_float(0.02, 0.04)
            learning_rate = quantum_lr_variation
        
        # Update emotions based on sentiment (slower, more gradual)
        self.emotional_state["joy"] += sentiment_score * learning_rate
        self.emotional_state["sadness"] -= sentiment_score * learning_rate
        
        # Add quantum randomness to curiosity (making responses more unpredictable)
        if self.quantum_random_available:
            quantum_curiosity_boost = self.get_quantum_random_float(-0.03, 0.03)
            self.emotional_state["curiosity"] = max(0.0, min(1.0, 
                self.emotional_state["curiosity"] + quantum_curiosity_boost))

        # Soft normalization - only normalize if emotions get too extreme
        # This prevents emotions from being forced to equal values
        total_emotion = sum(self.emotional_state.values())
        if total_emotion > 1.5 or total_emotion < 0.5:
            # Only normalize if emotions are way out of balance
            for emotion in self.emotional_state:
                self.emotional_state[emotion] = self.emotional_state[emotion] / total_emotion if total_emotion > 0 else 0.2
        else:
            # Just ensure no emotion goes below minimum threshold
            for emotion in self.emotional_state:
                if self.emotional_state[emotion] < 0.05:
                    self.emotional_state[emotion] = 0.05
        
        # Save to JSON after update
        self._save_to_json()
        
        logging.info(f"[EmotionalStateAgent] Updated emotional state: {self.emotional_state}")
        return self.emotional_state
    
    def get_state(self):
        """Get current emotional state"""
        return self.emotional_state.copy()
    
    def is_ready(self):
        """Check if agent is ready"""
        return True  # Emotional state is always ready

