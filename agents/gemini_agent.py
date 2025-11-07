"""Gemini Thinking Agent - responsible for thinking and analysis using Gemini"""
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG
from llm_wrapper import LLMWrapper

class GeminiThinkingAgent:
    """Agent responsible for thinking and analysis using Gemini"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG or {}
        self.gemini_available = False
        
        # Get model from config
        gemini_config = self.config.get('gemini', {}) if self.config else {}
        gemini_model = gemini_config.get('model', 'gemini-2.0-flash-exp')
        
        # Initialize LLM wrapper with the model
        self.llm_wrapper = LLMWrapper(gemini_model=gemini_model, config=self.config)
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini API availability"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
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
            
            # Get hyperparameters from config
            gemini_config = self.config.get('gemini', {}) if self.config else {}
            temperature = gemini_config.get('temperature', 0.5)
            max_tokens = gemini_config.get('max_tokens', 200)
            
            # Call Gemini model (model is set in wrapper initialization)
            try:
                thinking_result = self.llm_wrapper.call_gemini(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if thinking_result and len(thinking_result) > 0:
                    logging.info("[GeminiThinkingAgent] ✓ Thinking completed")
                    return thinking_result
                else:
                    logging.error("[GeminiThinkingAgent] Model returned empty result")
                    return None
            except Exception as e:
                logging.error(f"[GeminiThinkingAgent] Model {self.llm_wrapper.gemini_model} failed: {e}")
                return None
            
        except Exception as e:
            logging.error(f"[GeminiThinkingAgent] Error: {e}")
            return None
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.gemini_available

