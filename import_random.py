import random
import nltk
import os
import json
import yaml
from dotenv import load_dotenv
import logging
import requests
from litellm import completion
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Load model configuration from YAML
def load_model_config(config_path="models.yaml"):
    """Load model configuration from YAML file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"✓ Model configuration loaded from {config_path}")
            return config
        else:
            logging.warning(f"⚠ Model configuration file {config_path} not found, using defaults")
            return None
    except Exception as e:
        logging.error(f"✗ Error loading model configuration: {e}")
        return None

# Load configuration at module level
MODEL_CONFIG = load_model_config()

# Download NLTK data (only needs to be done once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt')
    
# Make sure punkt is downloaded before importing the rest
nltk.download('punkt', quiet=True)

# Import transformers with error handling
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    logging.warning("Transformers library not available. Using fallback sentiment analysis.")
    transformers_available = False

from enum import Enum

# ChromaDB removed - using JSON-only memory

# --- Memory System (JSON only) ---
class MemorySystem:
    """Memory system using JSON for simple key-value storage"""
    
    def __init__(self, json_db_path=None, config=None):
        self.config = config or MODEL_CONFIG or {}
        # Get paths from config or use defaults
        memory_config = self.config.get('memory', {}) if self.config else {}
        self.json_db_path = json_db_path or memory_config.get('json_path', './memory.json')
        self.json_memory = {}
        
        # Initialize JSON database
        self.load_json_memory()
    
    def is_ready(self):
        """Check if memory system is fully initialized"""
        return self.json_memory is not None
    
    def load_json_memory(self):
        """Load JSON memory database"""
        try:
            if os.path.exists(self.json_db_path):
                with open(self.json_db_path, 'r', encoding='utf-8') as f:
                    self.json_memory = json.load(f)
                logging.info(f"Loaded JSON memory with {len(self.json_memory)} entries")
            else:
                self.json_memory = {}
                logging.info("Created new JSON memory database")
        except Exception as e:
            logging.error(f"Error loading JSON memory: {e}")
            self.json_memory = {}
    
    def save_json_memory(self):
        """Save JSON memory database"""
        try:
            with open(self.json_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving JSON memory: {e}")
    
    def store_memory(self, text, metadata=None, memory_type="conversation"):
        """Store a memory in JSON"""
        timestamp = datetime.now().isoformat()
        
        # Store in JSON
        memory_id = f"{memory_type}_{timestamp}"
        self.json_memory[memory_id] = {
            "text": text,
            "metadata": metadata or {},
            "type": memory_type,
            "timestamp": timestamp
        }
        self.save_json_memory()
        logging.info(f"Stored memory in JSON: {memory_id[:20]}...")
    
    def retrieve_relevant_memories(self, query, n_results=5):
        """Retrieve relevant memories using keyword search in JSON"""
        relevant_memories = []
        
        # Simple keyword search in JSON
        if self.json_memory:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for memory_id, memory_data in self.json_memory.items():
                text_lower = memory_data.get("text", "").lower()
                text_words = set(text_lower.split())
                
                # Simple overlap check
                overlap = len(query_words & text_words)
                if overlap > 0:
                    relevant_memories.append({
                        "text": memory_data["text"],
                        "metadata": memory_data.get("metadata", {}),
                        "distance": 1.0 - (overlap / max(len(query_words), len(text_words)))
                    })
            
            # Sort by relevance (lower distance = more relevant)
            relevant_memories.sort(key=lambda x: x.get("distance", 1.0))
            relevant_memories = relevant_memories[:n_results]
            logging.info(f"Retrieved {len(relevant_memories)} relevant memories from JSON DB")
        
        return relevant_memories
    
    def get_json_memory(self, key):
        """Get a specific memory by key from JSON database"""
        return self.json_memory.get(key)
    
    def set_json_memory(self, key, value, metadata=None):
        """Set a key-value memory in JSON database"""
        self.json_memory[key] = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.save_json_memory()
    
    def get_all_json_memories(self):
        """Get all JSON memories"""
        return self.json_memory.copy()

# --- Agent Classes ---
class MemoryAgent:
    """Agent responsible for memory retrieval and storage"""
    
    def __init__(self, memory_system, config=None):
        self.memory_system = memory_system
        self.config = config or MODEL_CONFIG or {}
    
    def retrieve_memories(self, query, n_results=None):
        """Retrieve relevant memories for a query"""
        if n_results is None:
            max_memories = self.config.get('memory', {}).get('retrieval', {}).get('max_retrieved_memories', 5) if self.config else 5
        else:
            max_memories = n_results
        
        try:
            memories = self.memory_system.retrieve_relevant_memories(query, n_results=max_memories)
            if memories:
                logging.info(f"[MemoryAgent] Retrieved {len(memories)} relevant memories")
            return memories
        except Exception as e:
            logging.error(f"[MemoryAgent] Error retrieving memories: {e}")
            return []
    
    def store_memory(self, text, metadata=None, memory_type="conversation"):
        """Store a memory"""
        try:
            self.memory_system.store_memory(text, metadata, memory_type)
            logging.info(f"[MemoryAgent] Stored memory: {memory_type}")
        except Exception as e:
            logging.error(f"[MemoryAgent] Error storing memory: {e}")
    
    def smoke_test(self):
        """Perform smoke test to verify memory system is working"""
        try:
            # Test storing
            test_text = "Smoke test memory entry"
            self.store_memory(test_text, {"test": True}, "test")
            
            # Test retrieving
            memories = self.retrieve_memories("smoke test", n_results=1)
            if memories is not None:
                logging.info("[MemoryAgent] ✓ Smoke test passed")
                return True
            else:
                logging.warning("[MemoryAgent] ⚠ Smoke test failed - retrieve returned None")
                return False
        except Exception as e:
            logging.error(f"[MemoryAgent] ✗ Smoke test failed: {e}")
            return False
    
    def is_ready(self):
        """Check if memory agent is ready"""
        return self.memory_system.is_ready() if self.memory_system else False

class GeminiThinkingAgent:
    """Agent responsible for thinking and analysis using Gemini"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.gemini_available = False
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini API availability"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            self.gemini_available = True
            logging.info("[GeminiThinkingAgent] ✓ Initialized and ready")
        else:
            logging.warning("[GeminiThinkingAgent] ✗ GEMINI_API_KEY not found")
    
    def think(self, user_input, emotional_state, conversation_history, retrieved_memories=None):
        """Think about and analyze the conversation context"""
        if not self.gemini_available:
            logging.warning("[GeminiThinkingAgent] Not available")
            return None
        
        try:
            # Build thinking prompt with conversation context
            emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])
            
            # Prepare conversation context for thinking
            context_summary = ""
            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 3 exchanges
                context_summary = "\nRecent conversation:\n"
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Galatea"
                    context_summary += f"{role}: {msg['content']}\n"
            
            # Add retrieved memories if available
            memory_context = ""
            if retrieved_memories and len(retrieved_memories) > 0:
                memory_context = "\n\nRelevant memories from past conversations:\n"
                for i, memory in enumerate(retrieved_memories[:3], 1):  # Top 3 most relevant
                    memory_context += f"{i}. {memory['text'][:200]}...\n"
            
            thinking_prompt = f"""You are the internal reasoning system for Galatea, an AI assistant.

Current emotional state: {emotions_text}
{context_summary}
{memory_context}
Current user message: "{user_input}"

Analyze this conversation and provide:
1. Key insights about what the user is asking or discussing
2. Important context from the conversation history and retrieved memories
3. How Galatea should respond emotionally and contextually
4. Any important details to remember or reference

Keep your analysis concise (2-3 sentences). Focus on what matters for crafting an appropriate response."""
            
            messages = [
                {"role": "system", "content": "You are an internal reasoning system. Analyze conversations and provide insights."},
                {"role": "user", "content": thinking_prompt}
            ]
            
            logging.info("[GeminiThinkingAgent] Processing thinking request...")
            
            # Get Gemini models from config
            gemini_config = self.config.get('gemini', {}) if self.config else {}
            gemini_models = gemini_config.get('thinking_models', [
                "gemini/gemini-2.0-flash-exp",
                "gemini/gemini-2.0-flash",
                "gemini/gemini-1.5-flash-latest",
                "gemini/gemini-1.5-flash"
            ])
            
            # Get thinking settings from config
            thinking_config = gemini_config.get('thinking', {})
            thinking_temp = thinking_config.get('temperature', 0.5)
            thinking_max_tokens = thinking_config.get('max_tokens', 200)
            
            for model in gemini_models:
                try:
                    response = completion(
                        model=model,
                        messages=messages,
                        temperature=thinking_temp,
                        max_tokens=thinking_max_tokens
                    )
                    
                    if response and 'choices' in response and len(response['choices']) > 0:
                        thinking_result = response['choices'][0]['message']['content']
                        logging.info("[GeminiThinkingAgent] ✓ Thinking completed")
                        return thinking_result.strip()
                except Exception as e:
                    logging.warning(f"[GeminiThinkingAgent] Model {model} failed: {e}, trying next...")
                    continue

            logging.error("[GeminiThinkingAgent] All models failed")
            return None
            
        except Exception as e:
            logging.error(f"[GeminiThinkingAgent] Error: {e}")
            return None
    
    def smoke_test(self):
        """Perform smoke test to verify Gemini is working"""
        if not self.gemini_available:
            return False
        
        try:
            test_result = self.think(
                "test",
                {"joy": 0.5, "sadness": 0.3, "anger": 0.1, "fear": 0.1, "curiosity": 0.5},
                [],
                retrieved_memories=None
            )
            if test_result and len(test_result) > 0:
                logging.info("[GeminiThinkingAgent] ✓ Smoke test passed")
                return True
            else:
                logging.warning("[GeminiThinkingAgent] ⚠ Smoke test failed - no result")
                return False
        except Exception as e:
            logging.error(f"[GeminiThinkingAgent] ✗ Smoke test failed: {e}")
            return False
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.gemini_available

class PiResponseAgent:
    """Agent responsible for generating human-facing responses using Pi-3.1"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.inflection_ai_available = False
        self.inflection_ai_api_key = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Inflection AI API availability"""
        inflection_key = os.getenv("INFLECTION_AI_API_KEY")
        if inflection_key:
            self.inflection_ai_api_key = inflection_key
            self.inflection_ai_available = True
            logging.info("[PiResponseAgent] ✓ Initialized and ready")
        else:
            logging.warning("[PiResponseAgent] ✗ INFLECTION_AI_API_KEY not found")
    
    def respond(self, user_input, emotional_state, thinking_context=None, conversation_history=None, retrieved_memories=None):
        """Generate response using Pi-3.1 with thinking context and emotional state"""
        if not self.inflection_ai_available:
            logging.warning("[PiResponseAgent] Not available")
            return None

        try:
            # Create context with emotional state
            emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])
            
            # Get endpoint and config from YAML
            inflection_config = self.config.get('inflection_ai', {}) if self.config else {}
            url = inflection_config.get('api_endpoint', 'https://api.inflection.ai/external/api/inference')
            model_config = inflection_config.get('model_config', 'Pi-3.1')
            
            headers = {
                "Authorization": f"Bearer {self.inflection_ai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Build comprehensive context with thinking insights, conversation history, and retrieved memories
            context_parts = []
            
            # Base system context
            base_context = f"You are Galatea, an AI assistant with the following emotional state: {emotions_text}. Respond in character as Galatea. Keep your response concise (under 50 words) and reflect your emotional state in your tone."
            
            # Add thinking context from Gemini if available
            if thinking_context:
                base_context += f"\n\nInternal analysis: {thinking_context}"
            
            # Add retrieved memories if available
            if retrieved_memories and len(retrieved_memories) > 0:
                memory_text = "\n\nRelevant context from past conversations:\n"
                for i, memory in enumerate(retrieved_memories[:3], 1):  # Top 3 most relevant
                    memory_text += f"{i}. {memory['text'][:150]}...\n"
                base_context += memory_text
            
            # Add conversation history context
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-4:]  # Last 2 exchanges
                history_text = "\n\nRecent conversation context:\n"
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "You (Galatea)"
                    history_text += f"{role}: {msg['content']}\n"
                base_context += history_text
            
            context_parts.append({
                "text": base_context,
                "type": "System"
            })
            
            # Add conversation history as context messages
            if conversation_history and len(conversation_history) > 4:
                # Add older messages as context (but not the most recent ones we already included)
                for msg in conversation_history[-8:-4]:
                    context_parts.append({
                        "text": msg["content"],
                        "type": "Human" if msg["role"] == "user" else "Assistant"
                    })
            
            # Add current user input
            context_parts.append({
                "text": user_input,
                "type": "Human"
            })
            
            data = {
                "context": context_parts,
                "config": model_config
            }

            logging.info("[PiResponseAgent] Sending request to Pi-3.1 API...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Extract the response text from the API response
                if isinstance(result, dict):
                    if 'output' in result:
                        text = result['output']
                    elif 'text' in result:
                        text = result['text']
                    elif 'response' in result:
                        text = result['response']
                    elif 'message' in result:
                        text = result['message']
                    else:
                        text = str(result)
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
                
                logging.info("[PiResponseAgent] ✓ Response received")
                return text.strip()
            else:
                logging.error(f"[PiResponseAgent] API returned status code {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logging.error(f"[PiResponseAgent] Error: {e}")
            return None
    
    def smoke_test(self):
        """Perform smoke test to verify Pi-3.1 is working"""
        if not self.inflection_ai_available:
            return False
        
        try:
            test_result = self.respond(
                "Hello",
                {"joy": 0.5, "sadness": 0.3, "anger": 0.1, "fear": 0.1, "curiosity": 0.5},
                thinking_context="Test thinking context",
                conversation_history=[],
                retrieved_memories=None
            )
            if test_result and len(test_result) > 0:
                logging.info("[PiResponseAgent] ✓ Smoke test passed")
                return True
            else:
                logging.warning("[PiResponseAgent] ⚠ Smoke test failed - no result")
                return False
        except Exception as e:
            logging.error(f"[PiResponseAgent] ✗ Smoke test failed: {e}")
            return False
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.inflection_ai_available

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
    
    def smoke_test(self):
        """Perform smoke test to verify emotional state system is working"""
        try:
            # Test quantum randomness if available
            if self.quantum_random_available:
                test_float = self.get_quantum_random_float(0.0, 1.0)
                if not isinstance(test_float, float) or test_float < 0.0 or test_float > 1.0:
                    logging.warning("[EmotionalStateAgent] ⚠ Smoke test failed - invalid quantum random")
                    return False
            
            # Test state update
            initial_state = self.get_state().copy()
            updated_state = self.update_with_sentiment(0.5)
            if updated_state and isinstance(updated_state, dict):
                logging.info("[EmotionalStateAgent] ✓ Smoke test passed")
                return True
            else:
                logging.warning("[EmotionalStateAgent] ⚠ Smoke test failed - invalid state")
                return False
        except Exception as e:
            logging.error(f"[EmotionalStateAgent] ✗ Smoke test failed: {e}")
            return False
    
    def is_ready(self):
        """Check if agent is ready"""
        return True  # Emotional state is always ready

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
    
    def smoke_test(self):
        """Perform smoke test to verify Azure Text Analytics is working"""
        if not self.azure_available:
            return False
        
        try:
            test_text = "This is a test message for sentiment analysis."
            result = self.analyze(test_text)
            if result is not None:
                logging.info("[AzureTextAnalyticsAgent] ✓ Smoke test passed")
                return True
            else:
                logging.warning("[AzureTextAnalyticsAgent] ⚠ Smoke test failed - analyze returned None")
                return False
        except Exception as e:
            logging.error(f"[AzureTextAnalyticsAgent] ✗ Smoke test failed: {e}")
            return False
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.azure_available

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
    
    def smoke_test(self):
        """Perform smoke test to verify sentiment analysis is working"""
        try:
            test_text = "I am happy and excited!"
            result = self.analyze(test_text)
            if result is not None and isinstance(result, (int, float)):
                logging.info("[SentimentAgent] ✓ Smoke test passed")
                return True
            else:
                logging.warning("[SentimentAgent] ⚠ Smoke test failed - invalid result")
                return False
        except Exception as e:
            logging.error(f"[SentimentAgent] ✗ Smoke test failed: {e}")
            return False
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.ready

# --- 1. AI Core ---
class GalateaAI:
    def __init__(self):
        # Load model configuration first
        self.config = MODEL_CONFIG or {}
        
        self.knowledge_base = {}
        self.response_model = "A generic response" #Place Holder for the ML model
        
        # Conversation history for context
        self.conversation_history = []  # List of {"role": "user"/"assistant", "content": "..."}
        # Get max history length from config or use default
        self.max_history_length = self.config.get('conversation', {}).get('max_history_length', 20)
        
        # Initialize memory system
        logging.info("Initializing memory system (JSON)...")
        try:
            self.memory_system = MemorySystem(config=self.config)
            self.memory_system_ready = self.memory_system.is_ready()
            if not self.memory_system_ready:
                raise Exception("Memory system failed to initialize")
            logging.info("✓ Memory system initialized")
        except Exception as e:
            logging.error(f"Failed to initialize memory system: {e}")
            self.memory_system_ready = False
            raise
        
        # Initialize agents
        logging.info("Initializing agents...")
        self.memory_agent = MemoryAgent(self.memory_system, config=self.config)
        self.gemini_agent = GeminiThinkingAgent(config=self.config)
        self.pi_agent = PiResponseAgent(config=self.config)
        self.emotional_agent = EmotionalStateAgent(config=self.config)
        self.sentiment_agent = SentimentAgent(config=self.config)
        
        # Track initialization status
        self.memory_system_ready = self.memory_agent.is_ready()
        self.sentiment_analyzer_ready = self.sentiment_agent.is_ready()
        self.models_ready = self.gemini_agent.is_ready() or self.pi_agent.is_ready()
        self.api_keys_valid = self.gemini_agent.is_ready() or self.pi_agent.is_ready()
        
        # Legacy compatibility
        self.gemini_available = self.gemini_agent.is_ready()
        self.inflection_ai_available = self.pi_agent.is_ready()
        self.quantum_random_available = self.emotional_agent.quantum_random_available
        
        logging.info("✓ All agents initialized")
    
    def _check_pre_initialization(self):
        """Check if components were pre-initialized by initialize_galatea.py"""
        # Check if ChromaDB directory exists and has collection
        chromadb_path = "./chroma_db"
        if os.path.exists(chromadb_path):
            try:
                import chromadb
                from chromadb.config import Settings
                vector_db = chromadb.PersistentClient(
                    path=chromadb_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                collection = vector_db.get_collection("galatea_memory")
                if collection:
                    logging.info("✓ Pre-initialized ChromaDB detected")
                    return True
            except Exception:
                pass
        
        # Check if JSON memory exists
        if os.path.exists("./memory.json"):
            logging.info("✓ Pre-initialized JSON memory detected")
            return True
        
        return False
    
    def is_fully_initialized(self):
        """Check if all components are fully initialized"""
        return (
            self.memory_system_ready and
            self.sentiment_analyzer_ready and
            self.models_ready and
            self.api_keys_valid
        )
    
    def get_initialization_status(self):
        """Get detailed initialization status"""
        smoke_tests = getattr(self, 'smoke_test_results', {})
        return {
            "memory_system": self.memory_system_ready,
            "sentiment_analyzer": self.sentiment_analyzer_ready,
            "models": self.models_ready,
            "api_keys": self.api_keys_valid,
            "gemini_available": self.gemini_agent.is_ready() if hasattr(self, 'gemini_agent') else False,
            "inflection_ai_available": self.pi_agent.is_ready() if hasattr(self, 'pi_agent') else False,
            "azure_text_analytics_available": self.sentiment_agent.azure_agent.is_ready() if hasattr(self, 'sentiment_agent') else False,
            "smoke_tests": smoke_tests,
            "fully_initialized": self.is_fully_initialized()
        }
    
    @property
    def emotional_state(self):
        """Get current emotional state from EmotionalStateAgent"""
        return self.emotional_agent.get_state() if hasattr(self, 'emotional_agent') else {"joy": 0.2, "sadness": 0.2, "anger": 0.2, "fear": 0.2, "curiosity": 0.2}
        
    def initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis with fallback options"""
        self.sentiment_analyzer_ready = False
        # Get sentiment model from config
        sentiment_model = self.config.get('sentiment', {}).get('primary_model', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english') if self.config else 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
        
        if transformers_available:
            try:
                logging.info("Attempting to initialize Hugging Face sentiment analyzer")
                # Try to initialize the pipeline with specific parameters
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model=sentiment_model
                )
                self.sentiment_analyzer_ready = True
                logging.info("✓ Hugging Face sentiment analyzer loaded successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Hugging Face sentiment analyzer: {e}")
                self.sentiment_analyzer = None
                # Still mark as ready since we have fallback
                self.sentiment_analyzer_ready = True
                logging.info("✓ Using fallback sentiment analyzer")
        else:
            self.sentiment_analyzer = None
            self.sentiment_analyzer_ready = True  # Fallback available
            logging.info("✓ Using fallback sentiment analyzer")
            
    def analyze_sentiment(self, text):
        # Use Hugging Face if available
        if self.sentiment_analyzer is not None:
            try:
                result = self.sentiment_analyzer(text)[0]
                sentiment = result['label']
                score = result['score']
                
                if sentiment == 'POSITIVE':
                    return score
                else:
                    return -score
            except Exception as e:
                logging.error(f"Error in sentiment analysis: {e}")
                # Fall back to simple analysis
        
        # Simple fallback sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'happy', 'joy', 'love', 'like', 'wonderful']
        negative_words = ['bad', 'terrible', 'sad', 'hate', 'dislike', 'awful', 'poor', 'angry']
        
        words = text.lower().split()
        sentiment_score = 0.0
        
        for word in words:
            if word in positive_words:
                sentiment_score += 0.2
            elif word in negative_words:
                sentiment_score -= 0.2
                
        return max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
        
    def initialize_litellm(self):
        """Initialize LiteLLM for unified model management"""
        self.gemini_available = False
        self.inflection_ai_available = False
        self.quantum_random_available = False
        self.models_ready = False
        self.api_keys_valid = False
        
        # Check for Gemini API key
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            self.gemini_available = True
            logging.info("✓ Gemini API key found - Gemini models available via LiteLLM")
        else:
            logging.warning("GEMINI_API_KEY not found - Gemini models unavailable")
        
        # Check for Inflection AI API key
        inflection_key = os.getenv("INFLECTION_AI_API_KEY")
        if inflection_key:
            self.inflection_ai_api_key = inflection_key
            self.inflection_ai_available = True
            logging.info("✓ Inflection AI API key found - Pi-3.1 model available")
        else:
            logging.warning("INFLECTION_AI_API_KEY not found - Pi-3.1 model unavailable")
        
        # Check for Quantum Random Numbers API key
        quantum_key = os.getenv("ANU_QUANTUM_API_KEY")
        if quantum_key:
            self.quantum_api_key = quantum_key
            self.quantum_random_available = True
            logging.info("✓ ANU Quantum Numbers API key found - Quantum randomness available")
        else:
            logging.warning("ANU_QUANTUM_API_KEY not found - Quantum randomness unavailable")
        
        # Verify API keys are valid (at least one model API key must be present)
        self.api_keys_valid = self.gemini_available or self.inflection_ai_available
        if self.api_keys_valid:
            logging.info("✓ API keys validated - at least one model API key is available")
        else:
            logging.error("✗ No valid API keys found - models unavailable")
        
        # Models are ready if at least one is available
        self.models_ready = self.gemini_available or self.inflection_ai_available
        if self.models_ready:
            logging.info("✓ Models ready for use")
        else:
            logging.warning("⚠ No models available")
    
    def get_quantum_random_numbers(self, length=None, number_type=None):
        """Fetch quantum random numbers from ANU Quantum Numbers API"""
        if not self.quantum_random_available:
            logging.warning("Quantum random numbers unavailable, using fallback")
            return None
        
        # Get defaults from config
        quantum_config = self.config.get('quantum', {}) if self.config else {}
        if length is None:
            length = quantum_config.get('default_length', 128)
        if number_type is None:
            number_type = quantum_config.get('default_type', 'uint8')
        
        try:
            url = quantum_config.get('api_endpoint', 'https://api.quantumnumbers.anu.edu.au')
            headers = {
                "x-api-key": self.quantum_api_key
            }
            params = {
                "length": length,
                "type": number_type
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and 'data' in result:
                    logging.info(f"✓ Retrieved {len(result['data'])} quantum random numbers")
                    return result['data']
                else:
                    logging.warning("Quantum API returned success but no data")
                    return None
            else:
                logging.error(f"Quantum API returned status code {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error fetching quantum random numbers: {e}")
            return None
    
    def get_quantum_random_float(self, min_val=0.0, max_val=1.0):
        """Get a quantum random float between min_val and max_val"""
        quantum_nums = self.get_quantum_random_numbers(length=1, number_type='uint8')
        if quantum_nums and len(quantum_nums) > 0:
            # Normalize uint8 (0-255) to float range
            normalized = quantum_nums[0] / 255.0
            return min_val + (max_val - min_val) * normalized
        # Fallback to regular random
        return random.uniform(min_val, max_val)
    
    def call_inflection_ai(self, user_input, emotional_state, thinking_context=None, conversation_history=None, retrieved_memories=None):
        """Call Inflection AI Pi-3.1 model API with conversation context, thinking insights, and retrieved memories"""
        if not self.inflection_ai_available:
            return None

        try:
            # Create context with emotional state
            emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])
            
            # Format the request according to Inflection AI API
            # Get endpoint and config from YAML
            inflection_config = self.config.get('inflection_ai', {}) if self.config else {}
            url = inflection_config.get('api_endpoint', 'https://api.inflection.ai/external/api/inference')
            model_config = inflection_config.get('model_config', 'Pi-3.1')
            
            headers = {
                "Authorization": f"Bearer {self.inflection_ai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Build comprehensive context with thinking insights, conversation history, and retrieved memories
            context_parts = []
            
            # Base system context
            base_context = f"You are Galatea, an AI assistant with the following emotional state: {emotions_text}. Respond in character as Galatea. Keep your response concise (under 50 words) and reflect your emotional state in your tone."
            
            # Add thinking context from Gemini if available
            if thinking_context:
                base_context += f"\n\nInternal analysis: {thinking_context}"
            
            # Add retrieved memories if available
            if retrieved_memories and len(retrieved_memories) > 0:
                memory_text = "\n\nRelevant context from past conversations:\n"
                for i, memory in enumerate(retrieved_memories[:3], 1):  # Top 3 most relevant
                    memory_text += f"{i}. {memory['text'][:150]}...\n"
                base_context += memory_text
            
            # Add conversation history context
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-4:]  # Last 2 exchanges
                history_text = "\n\nRecent conversation context:\n"
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "You (Galatea)"
                    history_text += f"{role}: {msg['content']}\n"
                base_context += history_text
            
            context_parts.append({
                "text": base_context,
                "type": "System"
            })
            
            # Add conversation history as context messages
            if conversation_history and len(conversation_history) > 4:
                # Add older messages as context (but not the most recent ones we already included)
                for msg in conversation_history[-8:-4]:
                    context_parts.append({
                        "text": msg["content"],
                        "type": "Human" if msg["role"] == "user" else "Assistant"
                    })
            
            # Add current user input
            context_parts.append({
                "text": user_input,
                "type": "Human"
            })
            
            data = {
                "context": context_parts,
                "config": model_config
            }

            logging.info("Sending request to Inflection AI Pi-3.1 API")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Extract the response text from the API response
                if isinstance(result, dict):
                    if 'output' in result:
                        text = result['output']
                    elif 'text' in result:
                        text = result['text']
                    elif 'response' in result:
                        text = result['response']
                    elif 'message' in result:
                        text = result['message']
                    else:
                        text = str(result)
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
                
                logging.info("Inflection AI response received successfully")
                return text.strip()
            else:
                logging.error(f"Inflection AI API returned status code {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logging.error(f"Error calling Inflection AI API: {e}")
            logging.error(f"Full error details: {type(e).__name__}: {str(e)}")
            return None

    def gemini_think(self, user_input, emotional_state, conversation_history, retrieved_memories=None):
        """Use Gemini to think about and analyze the conversation context with retrieved memories"""
        if not self.gemini_available:
            return None
        
        try:
            # Build thinking prompt with conversation context
            emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])
            
            # Prepare conversation context for thinking
            context_summary = ""
            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 3 exchanges
                context_summary = "\nRecent conversation:\n"
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Galatea"
                    context_summary += f"{role}: {msg['content']}\n"
            
            # Add retrieved memories if available
            memory_context = ""
            if retrieved_memories and len(retrieved_memories) > 0:
                memory_context = "\n\nRelevant memories from past conversations:\n"
                for i, memory in enumerate(retrieved_memories[:3], 1):  # Top 3 most relevant
                    memory_context += f"{i}. {memory['text'][:200]}...\n"
            
            thinking_prompt = f"""You are the internal reasoning system for Galatea, an AI assistant.

Current emotional state: {emotions_text}
{context_summary}
{memory_context}
Current user message: "{user_input}"

Analyze this conversation and provide:
1. Key insights about what the user is asking or discussing
2. Important context from the conversation history and retrieved memories
3. How Galatea should respond emotionally and contextually
4. Any important details to remember or reference

Keep your analysis concise (2-3 sentences). Focus on what matters for crafting an appropriate response."""
            
            messages = [
                {"role": "system", "content": "You are an internal reasoning system. Analyze conversations and provide insights."},
                {"role": "user", "content": thinking_prompt}
            ]
            
            logging.info("Using Gemini for thinking/analysis")
            
            # Get Gemini models from config
            gemini_config = self.config.get('gemini', {}) if self.config else {}
            gemini_models = gemini_config.get('thinking_models', [
                "gemini/gemini-2.0-flash-exp",
                "gemini/gemini-2.0-flash",
                "gemini/gemini-1.5-flash-latest",
                "gemini/gemini-1.5-flash"
            ])
            
            # Get thinking settings from config
            thinking_config = gemini_config.get('thinking', {})
            thinking_temp = thinking_config.get('temperature', 0.5)
            thinking_max_tokens = thinking_config.get('max_tokens', 200)
            
            for model in gemini_models:
                try:
                    response = completion(
                        model=model,
                        messages=messages,
                        temperature=thinking_temp,
                        max_tokens=thinking_max_tokens
                    )
                    
                    if response and 'choices' in response and len(response['choices']) > 0:
                        thinking_result = response['choices'][0]['message']['content']
                        logging.info("✓ Gemini thinking completed")
                        return thinking_result.strip()
                except Exception as e:
                    logging.warning(f"Gemini model {model} failed for thinking: {e}, trying next...")
                    continue

            logging.error("All Gemini models failed for thinking")
            return None
            
        except Exception as e:
            logging.error(f"Error in Gemini thinking: {e}")
            return None
    
    def update_conversation_history(self, user_input, assistant_response):
        """Update conversation history, maintaining max length"""
        # Add user message
        self.conversation_history.append({"role": "user", "content": user_input})
        # Add assistant response
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            # Keep the most recent messages
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def store_important_memory(self, user_input, assistant_response, intent, keywords):
        """Store important conversation snippets in memory system"""
        try:
            # Determine if this conversation is worth storing
            # Store if: question, contains important keywords, or is a significant exchange
            should_store = False
            memory_type = "conversation"
            
            if intent == "question":
                should_store = True
                memory_type = "question"
            elif len(keywords) > 3:  # Substantial conversation
                should_store = True
            elif any(keyword in ["remember", "important", "note", "save"] for keyword in keywords):
                should_store = True
                memory_type = "important"
            
            if should_store:
                # Create a memory entry combining user input and response
                memory_text = f"User: {user_input}\nGalatea: {assistant_response}"
                
                metadata = {
                    "intent": intent,
                    "keywords": keywords[:5],  # Top 5 keywords
                    "emotions": {k: round(v, 2) for k, v in self.emotional_state.items()}
                }
                
                # Store in memory system (both ChromaDB and JSON)
                self.memory_system.store_memory(
                    text=memory_text,
                    metadata=metadata,
                    memory_type=memory_type
                )
                logging.info(f"Stored important memory: {memory_type} - {user_input[:50]}...")
        except Exception as e:
            logging.error(f"Error storing memory: {e}")
    
    def is_thinking_mode(self, intent, user_input, keywords):
        """Determine if the request requires thinking mode (use Gemini for complex reasoning)"""
        # Always use thinking mode now - Gemini always thinks, Pi-3.1 always responds
        return True

    def process_input(self, user_input):
        """Process user input through the agent chain workflow: PHI(GEMINI(User inputs, read with past memory), emotionalstate)"""
        # Step 1: Analyze sentiment
        sentiment_score = self.sentiment_agent.analyze(user_input)
        
        # Step 2: Extract keywords and determine intent
        keywords = self.extract_keywords(user_input)
        intent = self.determine_intent(user_input)
        
        # Step 3: Update emotional state based on sentiment
        self.emotional_agent.update_with_sentiment(sentiment_score)
        current_emotional_state = self.emotional_agent.get_state()
        
        # Step 4: Retrieve memories
        retrieved_memories = self.memory_agent.retrieve_memories(user_input)
        
        # Step 5: Chain workflow: PHI(GEMINI(User inputs, read with past memory), emotionalstate)
        # Step 5a: GEMINI(User inputs, read with past memory)
        thinking_context = self.gemini_agent.think(
            user_input,
            current_emotional_state,
            self.conversation_history,
            retrieved_memories=retrieved_memories
        )
        
        # Step 5b: PHI(GEMINI result, emotionalstate)
        response = self.pi_agent.respond(
            user_input,
            current_emotional_state,
            thinking_context=thinking_context,
            conversation_history=self.conversation_history,
            retrieved_memories=retrieved_memories
        )
        
        # Fallback if Pi-3.1 is not available
        if not response and self.gemini_agent.is_ready():
            response = self._gemini_fallback_response(
                user_input, 
                current_emotional_state, 
                thinking_context,
                self.conversation_history
            )
        
        # If still no response, use fallback
        if not response:
            response = self._generate_fallback_response(intent, keywords, current_emotional_state, user_input)
        
        # Update conversation history
        if response:
            self.update_conversation_history(user_input, response)
            
            # Store important memories
            self._store_important_memory(user_input, response, intent, keywords)
        
        # Update knowledge base
        self.update_knowledge(keywords, user_input)
        
        return response
        return response

    def extract_keywords(self, text):
        try:
            # Try using NLTK's tokenizer
            tokens = nltk.word_tokenize(text)
            keywords = [word.lower() for word in tokens if word.isalnum()]
            return keywords
        except Exception:
            # Fall back to a simple split-based approach if NLTK fails
            words = text.split()
            # Clean up words (remove punctuation)
            keywords = [word.lower().strip('.,!?;:()[]{}""\'') for word in words]
            # Filter out empty strings
            keywords = [word for word in keywords if word and word.isalnum()]
            return keywords

    def determine_intent(self, text):
        # More comprehensive intent recognition (using keywords)
        text = text.lower()
        if "what" in text or "how" in text or "why" in text:
            return "question"
        elif "thank" in text:
            return "gratitude"
        elif "goodbye" in text or "bye" in text:
            return "farewell"
        else:
            return "statement"

    def _gemini_fallback_response(self, user_input, emotional_state, thinking_context, conversation_history):
        """Fallback response using Gemini directly"""
        try:
            logging.info("[GalateaAI] Using Gemini fallback for direct response")
            emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])

            # Build messages with conversation history
            messages = []
            # Get system prompts from config
            system_prompts = self.config.get('system_prompts', {}) if self.config else {}
            identity = system_prompts.get('galatea_identity', 'You are Galatea, an AI assistant with emotional awareness and memory.')
            style = system_prompts.get('response_style', 'Respond in character, keeping responses concise (under 50 words).')
            
            messages.append({
                "role": "system", 
                "content": f"{identity} Your emotional state: {emotions_text}. {style}"
            })
            
            # Get fallback settings from config
            gemini_config = self.config.get('gemini', {}) if self.config else {}
            fallback_config = gemini_config.get('fallback', {})
            max_history_exchanges = fallback_config.get('max_history_exchanges', 8)
            fallback_model = gemini_config.get('fallback_model', 'gemini/gemini-1.5-flash')
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-max_history_exchanges:]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user input
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Add thinking context if available
            if thinking_context:
                messages.append({
                    "role": "system",
                    "content": f"Internal analysis: {thinking_context}"
                })
            
            # Use quantum randomness for temperature
            base_temperature = fallback_config.get('temperature_base', 0.7)
            temp_range = fallback_config.get('temperature_variation_range', [0.0, 0.3])
            quantum_temp_variation = self.emotional_agent.get_quantum_random_float(temp_range[0], temp_range[1])
            temperature = base_temperature + quantum_temp_variation
            
            response = completion(
                model=fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=fallback_config.get('max_tokens', 150)
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                text = response['choices'][0]['message']['content']
                logging.info("[GalateaAI] ✓ Gemini fallback response received")
                return text.strip()
        except Exception as e:
            logging.error(f"[GalateaAI] Gemini fallback failed: {e}")
        
        return None
    
    def _generate_fallback_response(self, intent, keywords, emotional_state, original_input):
        """Generate final fallback response when all systems fail"""
        logging.info(f"[GalateaAI] Using final fallback response. Intent: {intent}, Keywords: {keywords[:5]}")

        # Determine which systems are not working
        unavailable_systems = []
        system_descriptions = {
            'inflection_ai': ('Pi-3.1', 'my conversation model'),
            'gemini': ('Gemini', 'my thinking model'),
            'quantum_random': ('Quantum Random Numbers API', 'my quantum randomness source'),
            'memory': ('Memory System', 'my memory system')
        }
        
        if not getattr(self, 'inflection_ai_available', False):
            unavailable_systems.append(system_descriptions['inflection_ai'])
        if not getattr(self, 'gemini_available', False):
            unavailable_systems.append(system_descriptions['gemini'])
        if not getattr(self, 'quantum_random_available', False):
            unavailable_systems.append(system_descriptions['quantum_random'])
        if not getattr(self, 'memory_system_ready', False):
            unavailable_systems.append(system_descriptions['memory'])
        
        # Generate natural, conversational error message
        if unavailable_systems:
            if len(unavailable_systems) == 1:
                system_name, system_desc = unavailable_systems[0]
                system_msg = f"{system_desc} ({system_name}) is not working right now"
            elif len(unavailable_systems) == 2:
                sys1_name, sys1_desc = unavailable_systems[0]
                sys2_name, sys2_desc = unavailable_systems[1]
                system_msg = f"{sys1_desc} ({sys1_name}) and {sys2_desc} ({sys2_name}) are not working"
            else:
                # For 3+ systems, list them naturally
                system_list = []
                for sys_name, sys_desc in unavailable_systems[:-1]:
                    system_list.append(f"{sys_desc} ({sys_name})")
                last_name, last_desc = unavailable_systems[-1]
                system_msg = f"{', '.join(system_list)}, and {last_desc} ({last_name}) are not working"
        else:
            system_msg = "some of my systems encountered an error"
        
        fallback_response = None
        if intent == "question":
            if "you" in keywords:
                fallback_response = f"I'm still learning about myself, but I'm having technical difficulties. {system_msg.capitalize()}. I apologize for the inconvenience."
            else:
                fallback_response = f"I'd love to help with that, but {system_msg}. Please check my system status or try again in a moment."
        elif intent == "gratitude":
            fallback_response = "You're welcome!"
        else:
            if unavailable_systems:
                fallback_response = f"I hear you, but {system_msg}. This might be due to missing API keys or network issues. Please check my configuration."
            else:
                fallback_response = "I hear you, though my full AI capabilities aren't active right now. Please check if my API keys are configured."
        
        # Update conversation history even for fallback
        if fallback_response:
            self.update_conversation_history(original_input, fallback_response)
        
        return fallback_response

    def update_knowledge(self, keywords, user_input):
        #for new key words remember them
        for keyword in keywords:
            if keyword not in self.knowledge_base:
                self.knowledge_base[keyword] = user_input


# --- 2. Dialogue Engine ---
class DialogueEngine:
    def __init__(self, ai_core):
        self.ai_core = ai_core
        self.last_user_message = ""

    def get_response(self, user_input):
        # Store the last message for sentiment analysis
        self.last_user_message = user_input
        
        ai_response = self.ai_core.process_input(user_input)
        styled_response = self.apply_style(ai_response, self.ai_core.emotional_state)
        return styled_response

    def apply_style(self, text, emotional_state):
        style = self.get_style(emotional_state)
        #selects styles based on emotions
        #add style to text
        styled_text = text # Remove the style suffix to make responses cleaner
        return styled_text

    def get_style(self, emotional_state):
        #determine style based on the state of the AI
        return "neutral"

# --- 3. Avatar Engine ---

class AvatarShape(Enum): #create shape types for the avatar
    CIRCLE = "Circle"
    TRIANGLE = "Triangle"
    SQUARE = "Square"

class AvatarEngine:
    def __init__(self):
        self.avatar_model = "Circle"  # Start with a basic shape
        self.expression_parameters = {}

    def update_avatar(self, emotional_state):
        # Map emotions to avatar parameters (facial expressions, color)
        joy_level = emotional_state["joy"]
        sadness_level = emotional_state["sadness"]

        # Simple mapping (placeholder)
        self.avatar_model = self.change_avatar_shape(joy_level, sadness_level)

    def change_avatar_shape(self, joy, sad):
        #determine shape based on feelings
        if joy > 0.5:
            return AvatarShape.CIRCLE.value
        elif sad > 0.5:
            return AvatarShape.TRIANGLE.value
        else:
            return AvatarShape.SQUARE.value
            
    def render_avatar(self):
        # Simple console rendering of the avatar state
        print(f"Avatar shape: {self.avatar_model}")

# REMOVE THE MAIN PROGRAM LOOP THAT BLOCKS EXECUTION
# This is critical - the code below was causing the issue
# by creating instances outside of the Flask app's control

# instead, only run this if the script is executed directly
if __name__ == "__main__":
    # Download NLTK data again before starting the main loop to ensure availability
    nltk.download('punkt', quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download('punkt')

    #Create
    galatea_ai = GalateaAI()
    dialogue_engine = DialogueEngine(galatea_ai)
    avatar_engine = AvatarEngine()
    avatar_engine.update_avatar(galatea_ai.emotional_state)
    # Initial avatar rendering
    avatar_engine.render_avatar()

    while True:
        user_input = input("You: ")
        response = dialogue_engine.get_response(user_input)
        print(f"Galatea: {response}")

        avatar_engine.update_avatar(galatea_ai.emotional_state)
        avatar_engine.render_avatar()