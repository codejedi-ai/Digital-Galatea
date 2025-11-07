import random
import nltk
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

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

# --- 1. AI Core ---
class GalateaAI:
    def __init__(self):
        self.emotional_state = {"joy": 0.2, "sadness": 0.2, "anger": 0.2, "fear": 0.2, "curiosity": 0.2}
        self.knowledge_base = {}
        self.learning_rate = 0.05 # Reduced learning rate
        self.response_model = "A generic response" #Place Holder for the ML model
        
        # Initialize sentiment analyzer with fallback
        self.initialize_sentiment_analyzer()
        
        # Initialize Gemini API
        self.initialize_gemini()
        
    def initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis with fallback options"""
        if transformers_available:
            try:
                logging.info("Attempting to initialize Hugging Face sentiment analyzer")
                # Try to initialize the pipeline with specific parameters
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                )
                logging.info("Hugging Face sentiment analyzer loaded successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Hugging Face sentiment analyzer: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
            
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
        
    def initialize_gemini(self):
        """Initialize the Gemini API with API key from .env file"""
        self.gemini_available = False  # Default to False

        try:
            # Get API key from environment variable
            api_key = os.getenv("GEMINI_API_KEY")

            logging.info(f"Checking for GEMINI_API_KEY... Found: {bool(api_key)}")

            if not api_key:
                # Log error and fail gracefully (no input prompt for web deployment)
                logging.error("GEMINI_API_KEY not found in environment variables.")
                logging.error("Please set GEMINI_API_KEY in your environment or .env file")
                logging.error("Bot will use fallback responses only.")
                return

            # Configure the Gemini API
            logging.info("Configuring Gemini API...")
            genai.configure(api_key=api_key)
            
            # Use Gemini 2.0 Flash (latest and fastest model)
            try:
                # Try Gemini 2.0 Flash first (newest model)
                preferred_models = [
                    "gemini-2.0-flash-exp",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-flash",
                ]

                model_name = None
                last_error = None

                for model in preferred_models:
                    try:
                        logging.info(f"Attempting to initialize with model: {model}")
                        self.gemini_model = genai.GenerativeModel(model)

                        # Test the model with a simple prompt
                        logging.info(f"Testing {model} with a simple prompt...")
                        test_response = self.gemini_model.generate_content("Hello")

                        if hasattr(test_response, 'text') and test_response.text:
                            logging.info(f"✓ Test response received: {test_response.text[:50]}...")
                            model_name = model
                            self.gemini_available = True
                            logging.info(f"✓ Gemini API initialized successfully with model: {model_name}")
                            print(f"✓ Gemini API initialized successfully with model: {model_name}")
                            break
                        else:
                            logging.warning(f"Model {model} returned empty response")
                            continue

                    except Exception as e:
                        last_error = e
                        logging.warning(f"Model {model} failed: {e}")
                        continue

                if not model_name:
                    raise Exception(f"All models failed. Last error: {last_error}")

            except Exception as e:
                logging.error(f"Error initializing Gemini model: {e}")
                logging.error(f"Full error: {type(e).__name__}: {str(e)}")
                logging.error("Bot will use fallback responses only.")
                self.gemini_available = False

        except Exception as e:
            logging.error(f"Failed to initialize Gemini API: {e}")
            logging.error("Bot will use fallback responses only.")
            print(f"✗ Failed to initialize Gemini API: {e}")
            self.gemini_available = False

    def process_input(self, user_input):
        sentiment_score = self.analyze_sentiment(user_input)
        keywords = self.extract_keywords(user_input)
        intent = self.determine_intent(user_input)

        # Enhanced Emotion Update (decay and normalization)
        for emotion in self.emotional_state:
            # Decay emotions (more realistic fading)
            self.emotional_state[emotion] *= 0.9  # Decay by 10% each turn
            # Normalize
            self.emotional_state[emotion] = max(0.0, min(1.0, self.emotional_state[emotion]))

        self.emotional_state["joy"] += sentiment_score * self.learning_rate
        self.emotional_state["sadness"] -= sentiment_score * self.learning_rate

        # Re-normalize
        total_emotion = sum(self.emotional_state.values())
        for emotion in self.emotional_state:
            self.emotional_state[emotion] = self.emotional_state[emotion] / total_emotion if total_emotion > 0 else 0.2

        self.update_knowledge(keywords, user_input)
        response = self.generate_response(intent, keywords, self.emotional_state, user_input)
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

    def generate_response(self, intent, keywords, emotional_state, original_input):
        # Try to use Gemini API if available
        if hasattr(self, 'gemini_available') and self.gemini_available:
            try:
                # Create a prompt that includes emotional context and intent
                emotions_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])

                # Create a character prompt for Gemini
                prompt = f"""
                You are Galatea, an AI assistant with the following emotional state:
                {emotions_text}

                User input: "{original_input}"

                Respond in character as Galatea. Keep your response concise (under 50 words) and reflect your emotional state in your tone.
                If you're feeling more joy, be more enthusiastic. If sad, be more melancholic.
                """

                logging.info("Sending request to Gemini API")
                # Get response from Gemini with safety settings
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40
                }

                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                # Check if response is valid and return it
                if response and hasattr(response, 'text'):
                    logging.info(f"Gemini response received successfully")
                    return response.text.strip()
                elif hasattr(response, 'parts'):
                    # Try alternate access method
                    logging.info(f"Gemini response received via parts")
                    return response.parts[0].text.strip()
                else:
                    logging.warning(f"Unexpected response format: {response}")
                    # Fall back to basic response
                    return "I'm processing that..."

            except Exception as e:
                logging.error(f"Error using Gemini API: {e}")
                logging.error(f"Full error details: {type(e).__name__}: {str(e)}")
                print(f"Error using Gemini API: {e}")
                # Fall back to basic response logic
        else:
            logging.warning("Gemini API not available - using fallback responses")

        # Original response generation logic as fallback
        logging.info(f"Using fallback response. Intent: {intent}, Keywords: {keywords[:5]}")

        if intent == "question":
            if "you" in keywords:
                return "I am still learning about myself. My Gemini AI is not responding right now."
            else:
                return "I'd love to help with that, but my AI system isn't responding at the moment."
        elif intent == "gratitude":
            return "You're welcome!"
        else:
            return "I hear you, though my full AI capabilities aren't active right now. Please check if my API key is configured."

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