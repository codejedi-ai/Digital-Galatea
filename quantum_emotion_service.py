"""Quantum Emotion Service - Background service for quantum-influenced emotion updates"""
import os
import sys
import time
import logging
import requests
import threading
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIG

class QuantumEmotionService:
    """Background service that collects quantum random numbers and applies them to emotions"""
    
    def __init__(self, emotional_agent, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.emotional_agent = emotional_agent
        self.quantum_api_key = os.getenv("ANU_QUANTUM_API_KEY")
        self.running = False
        self.thread = None
        self.quantum_numbers = deque(maxlen=6)  # Store last 6 numbers
        self.last_update_time = time.time()
        self.update_interval = 10.0  # Update emotions every 10 seconds
        self.call_interval = 1.0  # Call API once per second
        
        if not self.quantum_api_key:
            logging.info("[QuantumEmotionService] Quantum API key not found - service will not run")
            return
        
        quantum_config = self.config.get('quantum', {}) if self.config else {}
        self.api_endpoint = quantum_config.get('api_endpoint', 'https://api.quantumnumbers.anu.edu.au')
        
        logging.info("[QuantumEmotionService] Initialized - will call API once per second")
    
    def _fetch_quantum_number(self):
        """Fetch a single quantum random number from the API"""
        if not self.quantum_api_key:
            return None
        
        try:
            headers = {"x-api-key": self.quantum_api_key}
            params = {"length": 1, "type": "uint8"}
            
            response = requests.get(self.api_endpoint, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and 'data' in result and len(result['data']) > 0:
                    # Normalize to 0-1 range
                    normalized = result['data'][0] / 255.0
                    return normalized
            elif response.status_code == 429:
                # Rate limited - return None, will use pseudo-random
                logging.debug("[QuantumEmotionService] Rate limited, using pseudo-random")
                return None
            else:
                logging.debug(f"[QuantumEmotionService] API returned {response.status_code}")
                return None
        except Exception as e:
            logging.debug(f"[QuantumEmotionService] API call failed: {e}")
            return None
        
        return None
    
    def _apply_quantum_emotions(self):
        """Apply the collected 6 quantum numbers to influence emotion state"""
        if len(self.quantum_numbers) < 6:
            logging.debug(f"[QuantumEmotionService] Not enough numbers yet ({len(self.quantum_numbers)}/6)")
            return
        
        # Convert deque to list
        numbers = list(self.quantum_numbers)
        
        # Map 6 numbers to 5 emotions + overall variation
        # numbers[0] -> joy
        # numbers[1] -> sadness
        # numbers[2] -> anger
        # numbers[3] -> fear
        # numbers[4] -> curiosity
        # numbers[5] -> overall variation factor
        
        emotions = ["joy", "sadness", "anger", "fear", "curiosity"]
        current_state = self.emotional_agent.get_state()
        
        # Apply quantum influence (subtle changes, -0.05 to +0.05 range)
        for i, emotion in enumerate(emotions):
            if i < len(numbers) and numbers[i] is not None:
                # Convert 0-1 to -0.05 to +0.05 influence
                influence = (numbers[i] - 0.5) * 0.1  # Scale to ±0.05
                
                # Apply with overall variation factor
                if len(numbers) > 5 and numbers[5] is not None:
                    variation = (numbers[5] - 0.5) * 0.02  # Additional ±0.01 variation
                    influence += variation
                
                # Update emotion
                new_value = current_state[emotion] + influence
                current_state[emotion] = max(0.05, min(1.0, new_value))
        
        # Soft normalization to keep emotions balanced (only if they get too extreme)
        total = sum(current_state.values())
        if total > 1.5 or total < 0.5:
            # Only normalize if emotions are way out of balance
            for emotion in emotions:
                current_state[emotion] = current_state[emotion] / total if total > 0 else 0.2
        else:
            # Just ensure no emotion goes below minimum threshold
            for emotion in emotions:
                if current_state[emotion] < 0.05:
                    current_state[emotion] = 0.05
        
        # Update the emotional agent's state
        self.emotional_agent.emotional_state = current_state
        
        # Save to JSON after quantum update
        self.emotional_agent._save_to_json()
        
        logging.info(f"[QuantumEmotionService] Applied quantum influence: {current_state}")
    
    def _service_loop(self):
        """Main service loop - calls API once per second, updates emotions every 10 seconds"""
        logging.info("[QuantumEmotionService] Service started")
        
        while self.running:
            try:
                # Fetch quantum number (once per second)
                quantum_num = self._fetch_quantum_number()
                
                if quantum_num is not None:
                    self.quantum_numbers.append(quantum_num)
                    logging.debug(f"[QuantumEmotionService] Collected quantum number: {quantum_num:.4f}")
                else:
                    # Use pseudo-random as fallback
                    import random
                    pseudo_random = random.random()
                    self.quantum_numbers.append(pseudo_random)
                    logging.debug(f"[QuantumEmotionService] Using pseudo-random: {pseudo_random:.4f}")
                
                # Check if it's time to update emotions (every 10 seconds)
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    if len(self.quantum_numbers) >= 6:
                        self._apply_quantum_emotions()
                        self.last_update_time = current_time
                        logging.info(f"[QuantumEmotionService] Updated emotions with {len(self.quantum_numbers)} quantum numbers")
                
                # Wait 1 second before next call
                time.sleep(self.call_interval)
                
            except Exception as e:
                logging.error(f"[QuantumEmotionService] Error in service loop: {e}")
                time.sleep(self.call_interval)  # Wait before retrying
    
    def start(self):
        """Start the background service"""
        if not self.quantum_api_key:
            logging.info("[QuantumEmotionService] Cannot start - no API key")
            return False
        
        if self.running:
            logging.warning("[QuantumEmotionService] Already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._service_loop, daemon=True)
        self.thread.start()
        logging.info("[QuantumEmotionService] Background service started")
        return True
    
    def stop(self):
        """Stop the background service"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info("[QuantumEmotionService] Background service stopped")
    
    def is_running(self):
        """Check if service is running"""
        return self.running

