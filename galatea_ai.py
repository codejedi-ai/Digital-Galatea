"""Main GalateaAI class - orchestrates all agents"""
import os
import sys
import nltk
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIG
from systems import MemorySystem
from agents import (
    MemoryAgent, DeepSeekThinkingAgent, PiResponseAgent,
    EmotionalStateAgent, SentimentAgent
)

# Download NLTK data (only needs to be done once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt')
    
# Make sure punkt is downloaded before importing the rest
nltk.download('punkt', quiet=True)

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
        self.deepseek_agent = DeepSeekThinkingAgent(config=self.config)
        self.pi_agent = PiResponseAgent(config=self.config)
        self.emotional_agent = EmotionalStateAgent(config=self.config)
        self.sentiment_agent = SentimentAgent(config=self.config)
        
        # Track initialization status
        self.memory_system_ready = self.memory_agent.is_ready()
        self.sentiment_analyzer_ready = self.sentiment_agent.is_ready()
        self.models_ready = self.deepseek_agent.is_ready() or self.pi_agent.is_ready()
        self.api_keys_valid = self.deepseek_agent.is_ready() or self.pi_agent.is_ready()
        
        # CRITICAL: Verify all critical systems are ready, raise exception if not
        if not self.memory_system_ready:
            raise RuntimeError("Memory system failed to initialize - application cannot continue")
        if not self.sentiment_analyzer_ready:
            raise RuntimeError("Sentiment analyzer failed to initialize - application cannot continue")
        if not self.models_ready:
            raise RuntimeError("No AI models available (DeepSeek or Pi-3.1) - application cannot continue")
        if not self.api_keys_valid:
            raise RuntimeError("API keys are invalid or missing - application cannot continue")
        if not self.pi_agent.is_ready():
            raise RuntimeError("Pi-3.1 (PHI) model is not available - application cannot continue")
        if not self.deepseek_agent.is_ready():
            raise RuntimeError("DeepSeek model is not available - application cannot continue")
        
        # Legacy compatibility
        self.deepseek_available = self.deepseek_agent.is_ready()
        self.inflection_ai_available = self.pi_agent.is_ready()
        self.quantum_random_available = self.emotional_agent.quantum_random_available
        
        logging.info("✓ All agents initialized and verified")
    
    def _check_pre_initialization(self):
        """Check if components were pre-initialized by initialize_galatea.py"""
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
        return {
            "memory_system": self.memory_system_ready,
            "sentiment_analyzer": self.sentiment_analyzer_ready,
            "models": self.models_ready,
            "api_keys": self.api_keys_valid,
            "deepseek_available": self.deepseek_agent.is_ready() if hasattr(self, 'deepseek_agent') else False,
            "inflection_ai_available": self.pi_agent.is_ready() if hasattr(self, 'pi_agent') else False,
            "azure_text_analytics_available": self.sentiment_agent.azure_agent.is_ready() if hasattr(self, 'sentiment_agent') else False,
            "fully_initialized": self.is_fully_initialized()
        }
    
    @property
    def emotional_state(self):
        """Get current emotional state from EmotionalStateAgent"""
        return self.emotional_agent.get_state() if hasattr(self, 'emotional_agent') else {"joy": 0.2, "sadness": 0.2, "anger": 0.2, "fear": 0.2, "curiosity": 0.2}
    
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
    
    def _store_important_memory(self, user_input, assistant_response, intent, keywords):
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
                    "emotions": {k: round(v, 2) for k, v in self.emotional_agent.get_state().items()}
                }
                
                # Store in memory system
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
        
        # Step 5: Chain workflow: PHI(DEEPSEEK(User inputs, read with past memory), emotionalstate)
        # Step 5a: DEEPSEEK(User inputs, read with past memory)
        thinking_context = self.deepseek_agent.think(
            user_input,
            current_emotional_state,
            self.conversation_history,
            retrieved_memories=retrieved_memories
        )
        
        # Step 5b: PHI(DEEPSEEK result, emotionalstate)
        response = self.pi_agent.respond(
            user_input,
            current_emotional_state,
            thinking_context=thinking_context,
            conversation_history=self.conversation_history,
            retrieved_memories=retrieved_memories
        )
        
        # CRITICAL: Pi-3.1 (PHI) model must generate response - raise exception if it fails
        if not response:
            error_msg = "[GalateaAI] CRITICAL: Pi-3.1 (PHI) model failed to generate response. Application cannot continue."
            logging.error("=" * 60)
            logging.error(error_msg)
            logging.error("=" * 60)
            raise RuntimeError(error_msg)
        
        # Update conversation history
        self.update_conversation_history(user_input, response)
        
        # Store important memories
        self._store_important_memory(user_input, response, intent, keywords)
        
        # Update knowledge base
        self.update_knowledge(keywords, user_input)
        
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


    def update_knowledge(self, keywords, user_input):
        #for new key words remember them
        for keyword in keywords:
            if keyword not in self.knowledge_base:
                self.knowledge_base[keyword] = user_input

