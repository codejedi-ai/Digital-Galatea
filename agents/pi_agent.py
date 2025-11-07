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
            
            # Build comprehensive context - Inflection AI API only accepts "Human" and "Assistant" types
            # We'll incorporate system instructions into the first Human message
            context_parts = []
            
            # Build system instructions as part of the user input context
            system_instructions = f"[Context: You are Galatea, an AI assistant. Emotional state: {emotions_text}. "
            
            # Add thinking context from Gemini if available
            if thinking_context:
                system_instructions += f"Internal analysis: {thinking_context}. "
            
            # Add retrieved memories if available
            if retrieved_memories and len(retrieved_memories) > 0:
                memory_text = "Relevant memories: "
                for i, memory in enumerate(retrieved_memories[:3], 1):  # Top 3 most relevant
                    memory_text += f"{i}. {memory['text'][:100]}; "
                system_instructions += memory_text
            
            system_instructions += "Keep response concise (under 50 words) and reflect emotional state.]"
            
            # Add conversation history as context messages (Human/Assistant only)
            if conversation_history and len(conversation_history) > 0:
                # Include recent conversation history
                for msg in conversation_history[-6:]:  # Last 3 exchanges (6 messages)
                    context_parts.append({
                        "text": msg["content"],
                        "type": "Human" if msg["role"] == "user" else "Assistant"
                    })
            
            # Add current user input with system context prepended
            enhanced_user_input = f"{system_instructions}\n\n{user_input}"
            context_parts.append({
                "text": enhanced_user_input,
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

