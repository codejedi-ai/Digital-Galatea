"""Pi Response Agent - responsible for generating human-facing responses using Pi-3.1"""
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG
from llm_wrapper import LLMWrapper

class PiResponseAgent:
    """Agent responsible for generating human-facing responses using Pi-3.1"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.inflection_ai_available = False
        
        # Get model from config
        inflection_config = self.config.get('inflection_ai', {}) if self.config else {}
        inflection_model = inflection_config.get('model', 'Pi-3.1')
        
        # Initialize LLM wrapper with the model
        self.llm_wrapper = LLMWrapper(inflection_model=inflection_model, config=self.config)
        self._initialize()
    
    def _initialize(self):
        """Initialize Inflection AI API availability"""
        inflection_key = os.getenv("INFLECTION_AI_API_KEY")
        if inflection_key:
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
            
            # Build comprehensive context with thinking insights, conversation history, and retrieved memories
            context_parts = []
            
            # Base system context - Inflection AI uses lowercase "system" type
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
                "type": "Instruction"  # Use "Instruction" type for system instructions (System requires event_type)
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

            logging.info("[PiResponseAgent] Sending request to Pi-3.1 API...")
            # Model is set in wrapper initialization
            response = self.llm_wrapper.call_inflection_ai(context_parts)
            
            if response:
                logging.info("[PiResponseAgent] ✓ Response received")
                return response
            else:
                logging.error("[PiResponseAgent] API call failed")
                return None

        except Exception as e:
            logging.error(f"[PiResponseAgent] Error: {e}")
            return None
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.inflection_ai_available

