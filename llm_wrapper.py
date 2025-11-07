"""Custom LLM Wrapper - Direct API calls using requests (no LiteLLM)"""
import os
import sys
import logging
import requests

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIG

class LLMWrapper:
    """Custom LLM wrapper for Gemini and Inflection AI using direct API calls"""
    
    def __init__(self, gemini_model=None, inflection_model=None, config=None):
        """
        Initialize LLM Wrapper with models and configuration
        
        Args:
            gemini_model: Gemini model name (e.g., 'gemini-2.0-flash-exp')
            inflection_model: Inflection AI model name (e.g., 'Pi-3.1')
            config: Configuration dict (optional, will load from MODEL_CONFIG if not provided)
        """
        self.config = config or MODEL_CONFIG or {}
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.inflection_ai_api_key = os.getenv("INFLECTION_AI_API_KEY")
        
        # Set models from parameters or config
        if gemini_model:
            self.gemini_model = gemini_model
        else:
            gemini_config = self.config.get('gemini', {}) if self.config else {}
            self.gemini_model = gemini_config.get('model', 'gemini-2.0-flash-exp')
        
        if inflection_model:
            self.inflection_model = inflection_model
        else:
            inflection_config = self.config.get('inflection_ai', {}) if self.config else {}
            self.inflection_model = inflection_config.get('model', 'Pi-3.1')
        
        # Remove 'gemini/' prefix if present
        if self.gemini_model.startswith('gemini/'):
            self.gemini_model = self.gemini_model.replace('gemini/', '')
        
        logging.info(f"[LLMWrapper] Initialized with Gemini model: {self.gemini_model}, Inflection model: {self.inflection_model}")
    
    def call_gemini(self, messages, temperature=0.7, max_tokens=1024):
        """
        Call Gemini API directly using requests
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response text or None if failed
        """
        if not self.gemini_api_key:
            logging.error("[LLMWrapper] GEMINI_API_KEY not found")
            return None
        
        # Use the model set during initialization
        model = self.gemini_model
        
        # Gemini API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.gemini_api_key
        }
        
        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                system_instruction = content
            elif role == 'user':
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == 'assistant':
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        # Add system instruction if present
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        try:
            logging.info(f"[LLMWrapper] Calling Gemini API: {model}")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract text from Gemini response
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            text = parts[0]['text']
                            logging.info("[LLMWrapper] ✓ Gemini response received")
                            return text.strip()
                
                logging.error(f"[LLMWrapper] Unexpected Gemini response format: {result}")
                return None
            else:
                logging.error(f"[LLMWrapper] Gemini API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"[LLMWrapper] Error calling Gemini API: {e}")
            return None
    
    def call_inflection_ai(self, context_parts):
        """
        Call Inflection AI API directly using requests
        
        Args:
            context_parts: List of context dicts with 'text' and 'type'
            
        Returns:
            Response text or None if failed
        """
        if not self.inflection_ai_api_key:
            logging.error("[LLMWrapper] INFLECTION_AI_API_KEY not found")
            return None
        
        # Use the model set during initialization
        model_config = self.inflection_model
        
        # Get endpoint from config
        inflection_config = self.config.get('inflection_ai', {}) if self.config else {}
        url = inflection_config.get('api_endpoint', 'https://api.inflection.ai/external/api/inference')
        
        headers = {
            "Authorization": f"Bearer {self.inflection_ai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "context": context_parts,
            "config": model_config
        }
        
        try:
            logging.info(f"[LLMWrapper] Calling Inflection AI API: {model_config}")
            logging.info(f"[LLMWrapper] Request URL: {url}")
            logging.info(f"[LLMWrapper] Request context parts count: {len(context_parts)}")
            logging.debug(f"[LLMWrapper] Request data: {data}")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            logging.info(f"[LLMWrapper] Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception as json_error:
                    logging.error(f"[LLMWrapper] Failed to parse JSON response: {json_error}")
                    logging.error(f"[LLMWrapper] Raw response text: {response.text[:500]}")
                    return None
                
                logging.info(f"[LLMWrapper] Response JSON: {result}")
                logging.info(f"[LLMWrapper] Response type: {type(result)}")
                logging.info(f"[LLMWrapper] Response keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                
                # Extract response text - Inflection AI returns text in 'text' field
                # Based on actual API response: {"created": ..., "text": "...", "tool_calls": [], "reasoning_content": null}
                text = None
                if isinstance(result, dict):
                    # Prioritize 'text' field as that's what the API actually returns
                    if 'text' in result:
                        text = result['text']
                        logging.debug(f"[LLMWrapper] Found text in 'text' field: {text[:100]}...")
                    elif 'output' in result:
                        text = result['output']
                        logging.debug(f"[LLMWrapper] Found text in 'output' field")
                    elif 'response' in result:
                        text = result['response']
                        logging.debug(f"[LLMWrapper] Found text in 'response' field")
                    elif 'message' in result:
                        text = result['message']
                        logging.debug(f"[LLMWrapper] Found text in 'message' field")
                    else:
                        # If result is a dict but no known field, try to get first string value
                        logging.warning(f"[LLMWrapper] No standard text field found, searching for string values...")
                        for key, value in result.items():
                            if isinstance(value, str) and value.strip():
                                text = value
                                logging.debug(f"[LLMWrapper] Found text in '{key}' field")
                                break
                        if not text:
                            logging.error(f"[LLMWrapper] No text found in response dict. Keys: {list(result.keys())}")
                            text = str(result)
                elif isinstance(result, str):
                    text = result
                    logging.debug(f"[LLMWrapper] Response is a string")
                else:
                    logging.warning(f"[LLMWrapper] Unexpected response type: {type(result)}")
                    text = str(result)
                
                if text and isinstance(text, str) and text.strip():
                    logging.info(f"[LLMWrapper] ✓ Inflection AI response received: {text[:100]}...")
                    return text.strip()
                else:
                    logging.error(f"[LLMWrapper] No valid text found in response. Text value: {text}, Type: {type(text)}")
                    logging.error(f"[LLMWrapper] Full response: {result}")
                    return None
            else:
                logging.error(f"[LLMWrapper] Inflection AI API returned status {response.status_code}")
                try:
                    error_detail = response.json()
                    logging.error(f"[LLMWrapper] Error details: {error_detail}")
                except:
                    logging.error(f"[LLMWrapper] Error response text: {response.text[:500]}")
                return None
                
        except Exception as e:
            logging.error(f"[LLMWrapper] Error calling Inflection AI API: {e}")
            return None

