from flask import Flask, render_template, request, jsonify, url_for
import os
import time
from dotenv import load_dotenv
import logging
from threading import Thread
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Debug: Log environment variable status
logging.info("=" * 60)
logging.info("ENVIRONMENT VARIABLES CHECK")
logging.info("=" * 60)
gemini_key = os.environ.get('GEMINI_API_KEY')
missing_gemini_key = False
if gemini_key:
    logging.info(f"✓ GEMINI_API_KEY found (length: {len(gemini_key)} chars)")
    logging.info(f"  First 10 chars: {gemini_key[:10]}...")
else:
    missing_gemini_key = True
    logging.error("=" * 60)
    logging.error("✗ GEMINI_API_KEY not found in environment!")
    logging.error("=" * 60)
    logging.error("")
    logging.error("The GEMINI_API_KEY environment variable is required for full functionality.")
    logging.error("")
    logging.error("For Hugging Face Spaces:")
    logging.error("  1. Go to Settings → Repository secrets")
    logging.error("  2. Click 'New secret'")
    logging.error("  3. Name: GEMINI_API_KEY")
    logging.error("  4. Value: [Your Google Gemini API key]")
    logging.error("  5. Get a key from: https://ai.google.dev/")
    logging.error("")
    logging.error("For local development:")
    logging.error("  1. Copy .env.example to .env")
    logging.error("  2. Add your API key to the .env file")
    logging.error("")
    logging.error("Available env vars starting with 'GEMINI': " +
                 str([k for k in os.environ.keys() if 'GEMINI' in k.upper()]))
    logging.error("Available env vars starting with 'GOOGLE': " +
                 str([k for k in os.environ.keys() if 'GOOGLE' in k.upper()]))
    logging.error("=" * 60)

logging.info("=" * 60)

# Download required NLTK data on startup
def download_nltk_data():
    """Download all required NLTK data for the application"""
    required_data = ['punkt', 'vader_lexicon']
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'sentiment/{data_name}.zip')
            logging.info(f"NLTK data '{data_name}' already downloaded")
        except LookupError:
            logging.info(f"Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)
            logging.info(f"Successfully downloaded NLTK data: {data_name}")

# Download NLTK data before app initialization
download_nltk_data()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables to hold components
galatea_ai = None
dialogue_engine = None
avatar_engine = None
is_initialized = False
initializing = False
gemini_initialized = False
max_init_retries = 3
current_init_retry = 0
init_script_running = False
init_script_complete = False

# Check for required environment variables
required_env_vars = ['GEMINI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logging.error("Please set these in your .env file or environment")
    print(f"⚠️ Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these in your .env file or environment")

def initialize_gemini():
    """Initialize Gemini API specifically"""
    global gemini_initialized
    
    if not galatea_ai:
        logging.warning("Cannot initialize Gemini: GalateaAI instance not created yet")
        return False
        
    if missing_gemini_key:
        logging.error("Cannot initialize Gemini: GEMINI_API_KEY is missing")
        return False

    try:
        # Check for GEMINI_API_KEY
        if not os.environ.get('GEMINI_API_KEY'):
            logging.error("GEMINI_API_KEY not found in environment variables")
            return False
            
        # Check if Gemini agent is ready (initialization happens automatically in GalateaAI.__init__)
        gemini_success = hasattr(galatea_ai, 'gemini_agent') and galatea_ai.gemini_agent.is_ready()
        
        if gemini_success:
            gemini_initialized = True
            logging.info("Gemini API initialized successfully")
            return True
        else:
            logging.error("Failed to initialize Gemini API")
            return False
    except Exception as e:
        logging.error(f"Error initializing Gemini API: {e}")
        return False

def run_init_script():
    """Run the initialization script in parallel"""
    global init_script_running, init_script_complete
    
    if init_script_running or init_script_complete:
        return
    
    init_script_running = True
    logging.info("=" * 70)
    logging.info("RUNNING PARALLEL INITIALIZATION SCRIPT")
    logging.info("=" * 70)
    
    try:
        import subprocess
        import sys
        
        # Run the initialization script
        script_path = os.path.join(os.path.dirname(__file__), 'initialize_galatea.py')
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logging.info("✓ Initialization script completed successfully")
            init_script_complete = True
        else:
            logging.error(f"✗ Initialization script failed with code {result.returncode}")
            logging.error(f"Error output: {result.stderr}")
            # Still mark as complete to allow app to continue
            init_script_complete = True
    except subprocess.TimeoutExpired:
        logging.error("✗ Initialization script timed out")
        init_script_complete = True
    except Exception as e:
        logging.error(f"✗ Error running initialization script: {e}")
        init_script_complete = True
    finally:
        init_script_running = False
        logging.info("=" * 70)

def initialize_components():
    """Initialize Galatea components (runs after init script completes)"""
    global galatea_ai, dialogue_engine, avatar_engine, is_initialized, initializing
    global current_init_retry, gemini_initialized, init_script_complete
    
    if initializing or is_initialized:
        return
    
    # Wait for initialization script to complete (poll every 2 seconds)
    max_wait_time = 300  # 5 minutes
    wait_start = time.time()
    while not init_script_complete:
        elapsed = time.time() - wait_start
        if elapsed > max_wait_time:
            logging.warning("Initialization script timeout - proceeding anyway")
            break
        logging.info(f"Waiting for initialization script to complete... ({elapsed:.0f}s)")
        time.sleep(2)
    
    if not init_script_complete:
        logging.warning("Proceeding with component initialization despite init script not completing")

    if missing_gemini_key:
        logging.error("Initialization aborted: GEMINI_API_KEY missing")
        return
        
    initializing = True
    logging.info("Starting to initialize Galatea components...")
    
    try:
        # Import here to avoid circular imports and ensure errors are caught
        from galatea_ai import GalateaAI
        from dialogue import DialogueEngine
        from avatar import AvatarEngine
        
        # Initialize components
        logging.info("=" * 60)
        logging.info("INITIALIZING GALATEA AI SYSTEM")
        logging.info("=" * 60)
        
        galatea_ai = GalateaAI()
        dialogue_engine = DialogueEngine(galatea_ai)
        avatar_engine = AvatarEngine()
        avatar_engine.update_avatar(galatea_ai.emotional_state)
        
        # Check if all components are fully initialized
        init_status = galatea_ai.get_initialization_status()
        
        logging.info("=" * 60)
        logging.info("INITIALIZATION STATUS")
        logging.info("=" * 60)
        logging.info(f"Memory System (JSON): {init_status['memory_system']}")
        logging.info(f"Sentiment Analyzer: {init_status['sentiment_analyzer']}")
        logging.info(f"Models Ready: {init_status['models']}")
        logging.info(f"  - Gemini available: {init_status['gemini_available']}")
        logging.info(f"  - Inflection AI available: {init_status['inflection_ai_available']}")
        logging.info(f"API Keys Valid: {init_status['api_keys']}")
        logging.info(f"Fully Initialized: {init_status['fully_initialized']}")
        logging.info("=" * 60)
        
        # CRITICAL: Only mark as initialized if ALL components are ready
        # If any component fails, EXIT the application immediately
        if init_status['fully_initialized']:
            is_initialized = True
            logging.info("✓ Galatea AI system fully initialized and ready")
            logging.info(f"Emotions initialized: {galatea_ai.emotional_state}")
        else:
            logging.error("=" * 60)
            logging.error("❌ INITIALIZATION FAILED - EXITING APPLICATION")
            logging.error("=" * 60)
            logging.error("One or more critical components failed to initialize:")
            if not init_status['memory_system']:
                logging.error("  ✗ Memory System (JSON) - FAILED")
            if not init_status['sentiment_analyzer']:
                logging.error("  ✗ Sentiment Analyzer - FAILED")
            if not init_status['models']:
                logging.error("  ✗ Models - FAILED")
            if not init_status['api_keys']:
                logging.error("  ✗ API Keys - FAILED")
            logging.error("=" * 60)
            logging.error("EXITING APPLICATION - All systems must be operational")
            logging.error("=" * 60)
            import sys
            sys.exit(1)  # Exit immediately - no retries, no partial functionality
    except Exception as e:
        logging.error("=" * 60)
        logging.error(f"❌ CRITICAL ERROR INITIALIZING GALATEA: {e}")
        logging.error("=" * 60)
        logging.error("EXITING APPLICATION - Cannot continue with initialization failure")
        logging.error("=" * 60)
        print(f"CRITICAL ERROR: {e}")
        print("Application exiting due to initialization failure")
        import sys
        sys.exit(1)  # Exit immediately - no retries
    finally:
        initializing = False

@app.route('/')
def home():
    # Add error handling for template rendering
    try:
        # Start initialization script in background if not already started
        if not init_script_complete and not init_script_running:
            Thread(target=run_init_script, daemon=True).start()
        
        # Start component initialization after init script (will wait if script not done)
        if not is_initialized and not initializing and not missing_gemini_key:
            Thread(target=initialize_components, daemon=True).start()
            
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        return f"Error loading the application: {e}. Make sure templates/index.html exists.", 500

@app.route('/api/chat', methods=['POST'])
def chat():
    # CRITICAL: Do not allow chat if system is not fully initialized
    if not is_initialized:
        return jsonify({
            'error': 'System is not initialized yet. Please wait for initialization to complete.',
            'is_initialized': False,
            'status': 'initializing'
        }), 503  # Service Unavailable
    
    # Check if API key is missing
    if missing_gemini_key:
        return jsonify({
            'error': 'GEMINI_API_KEY is missing. Chat is unavailable.',
            'status': 'missing_gemini_key',
            'is_initialized': False
        }), 503
    
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Process the message through Galatea
        response = dialogue_engine.get_response(user_input)
        
        # CRITICAL: If response is None, Pi-3.1 failed - exit application
        if response is None:
            error_msg = "CRITICAL: Pi-3.1 (PHI) model failed to generate response. Application cannot continue."
            logging.error("=" * 60)
            logging.error(error_msg)
            logging.error("=" * 60)
            import sys
            sys.exit(1)  # Exit immediately
        
        # Update avatar
        avatar_engine.update_avatar(galatea_ai.emotional_state)
        avatar_shape = avatar_engine.avatar_model
        
        # Get emotional state for frontend
        emotions = {k: round(v, 2) for k, v in galatea_ai.emotional_state.items()}
        
        logging.info(f"Chat response: {response}, avatar: {avatar_shape}, emotions: {emotions}")
        
        return jsonify({
            'response': response,
            'avatar_shape': avatar_shape,
            'emotions': emotions,
            'is_initialized': True
        })
    except RuntimeError as e:
        # CRITICAL: RuntimeError means a system failure - exit application
        error_msg = f"CRITICAL SYSTEM FAILURE: {e}"
        logging.error("=" * 60)
        logging.error(error_msg)
        logging.error("EXITING APPLICATION")
        logging.error("=" * 60)
        import sys
        sys.exit(1)  # Exit immediately
    except Exception as e:
        # Any other exception is also critical - exit application
        error_msg = f"CRITICAL ERROR processing chat: {e}"
        logging.error("=" * 60)
        logging.error(error_msg)
        logging.error("EXITING APPLICATION")
        logging.error("=" * 60)
        import sys
        sys.exit(1)  # Exit immediately

# Import Azure Text Analytics with fallback to NLTK VADER
try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    azure_available = True
except ImportError:
    azure_available = False
    logging.warning("Azure Text Analytics not installed, will use NLTK VADER for sentiment analysis")

# Set up NLTK VADER as fallback
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon on first run
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logging.info("Downloading NLTK VADER lexicon for offline sentiment analysis")
    nltk.download('vader_lexicon')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Azure Text Analytics client setup
def get_text_analytics_client():
    if not azure_available:
        return None
        
    key = os.environ.get("AZURE_TEXT_ANALYTICS_KEY")
    endpoint = os.environ.get("AZURE_TEXT_ANALYTICS_ENDPOINT")
    
    if not key or not endpoint:
        logging.warning("Azure Text Analytics credentials not found in environment variables")
        return None
        
    try:
        credential = AzureKeyCredential(key)
        client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
        return client
    except Exception as e:
        logging.error(f"Error creating Azure Text Analytics client: {e}")
        return None

# Analyze sentiment using Azure with VADER fallback
def analyze_sentiment(text):
    # Try Azure first
    client = get_text_analytics_client()
    if client and text:
        try:
            response = client.analyze_sentiment([text])[0]
            sentiment_scores = {
                "positive": response.confidence_scores.positive,
                "neutral": response.confidence_scores.neutral,
                "negative": response.confidence_scores.negative,
                "sentiment": response.sentiment
            }
            logging.info(f"Using Azure sentiment analysis: {sentiment_scores}")
            return sentiment_scores
        except Exception as e:
            logging.error(f"Error with Azure sentiment analysis: {e}")
            # Fall through to VADER
    
    # Fallback to NLTK VADER
    if text:
        try:
            scores = vader_analyzer.polarity_scores(text)
            # Map VADER scores to Azure-like format
            positive = scores['pos']
            negative = scores['neg']
            neutral = scores['neu']
            
            # Keywords that indicate anger
            anger_keywords = ["angry", "mad", "furious", "outraged", "annoyed", "irritated", 
                             "frustrated", "hate", "hatred", "despise", "resent", "enraged"]
            
            # Check for anger keywords
            has_anger = any(word in text.lower() for word in anger_keywords)
            
            # Determine overall sentiment with enhanced anger detection
            if scores['compound'] >= 0.05:
                sentiment = "positive"
            elif scores['compound'] <= -0.05:
                if has_anger:
                    sentiment = "angry"  # Special anger category
                else:
                    sentiment = "negative"
            else:
                sentiment = "neutral"
                
            sentiment_scores = {
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
                "angry": 1.0 if has_anger else 0.0,  # Add special anger score
                "sentiment": sentiment
            }
            logging.info(f"Using enhanced VADER sentiment analysis: {sentiment_scores}")
            return sentiment_scores
        except Exception as e:
            logging.error(f"Error with VADER sentiment analysis: {e}")
    
    return None

# Track avatar updates with timestamp
last_avatar_update = time.time()

@app.route('/api/avatar')
def get_avatar():
    """Endpoint to get the current avatar shape and state with enhanced responsiveness"""
    global last_avatar_update
    
    if not is_initialized:
        return jsonify({
            'avatar_shape': 'Circle',
            'is_initialized': False,
            'last_updated': last_avatar_update,
            'status': 'initializing'
        })
    
    try:
        avatar_shape = avatar_engine.avatar_model if avatar_engine else 'Circle'
        
        # Update timestamp when the avatar changes (you would track this in AvatarEngine normally)
        current_timestamp = time.time()
        
        # Get the last message for sentiment analysis if available
        last_message = getattr(dialogue_engine, 'last_user_message', '')
        sentiment_data = None
        
        # Analyze sentiment if we have a message
        if last_message:
            sentiment_data = analyze_sentiment(last_message)
            
        # Force avatar update based on emotions if available
        if avatar_engine and galatea_ai:
            # If we have sentiment data, incorporate it into emotional state
            if sentiment_data:
                # Update emotional state based on sentiment (enhanced mapping)
                if sentiment_data["sentiment"] == "positive":
                    galatea_ai.emotional_state["joy"] = max(galatea_ai.emotional_state["joy"], sentiment_data["positive"])
                elif sentiment_data["sentiment"] == "negative":
                    galatea_ai.emotional_state["sadness"] = max(galatea_ai.emotional_state["sadness"], sentiment_data["negative"])
                elif sentiment_data["sentiment"] == "angry":
                    # Amplify anger emotion when detected
                    galatea_ai.emotional_state["anger"] = max(galatea_ai.emotional_state["anger"], 0.8)
            
            avatar_engine.update_avatar(galatea_ai.emotional_state)
            avatar_shape = avatar_engine.avatar_model
            last_avatar_update = current_timestamp
            
        return jsonify({
            'avatar_shape': avatar_shape,
            'emotions': {k: round(v, 2) for k, v in galatea_ai.emotional_state.items()} if galatea_ai else {},
            'sentiment': sentiment_data,
            'is_initialized': is_initialized,
            'last_updated': last_avatar_update,
            'status': 'ready'
        })
    except Exception as e:
        logging.error(f"Error getting avatar: {e}")
        return jsonify({
            'error': 'Failed to get avatar information',
            'avatar_shape': 'Circle',
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Simple health check endpoint to verify the server is running"""
    return jsonify({
        'status': 'ok',
        'gemini_available': hasattr(galatea_ai, 'gemini_available') and galatea_ai.gemini_available if galatea_ai else False,
        'is_initialized': is_initialized,
        'missing_gemini_key': missing_gemini_key
    })

@app.route('/api/availability')
def availability():
    """Report overall availability state to the frontend"""
    if missing_gemini_key:
        return jsonify({
            'available': False,
            'status': 'missing_gemini_key',
            'is_initialized': False,
            'initializing': False,
            'missing_gemini_key': True,
            'error_page': url_for('error_page')
        })

    if initializing or not is_initialized:
        return jsonify({
            'available': False,
            'status': 'initializing',
            'is_initialized': is_initialized,
            'initializing': initializing,
            'missing_gemini_key': False
        })

    return jsonify({
        'available': True,
        'status': 'ready',
        'is_initialized': True,
        'initializing': False,
        'missing_gemini_key': False
    })

@app.route('/api/is_initialized')
def is_initialized_endpoint():
    """Lightweight endpoint for polling initialization progress"""
    global init_script_running, init_script_complete
    
    # Determine current initialization state
    if missing_gemini_key:
        return jsonify({
            'is_initialized': False,
            'initializing': False,
            'missing_gemini_key': True,
            'error_page': url_for('error_page'),
            'status': 'missing_api_key'
        })
    
    # Check if init script is still running
    if init_script_running:
        return jsonify({
            'is_initialized': False,
            'initializing': True,
            'missing_gemini_key': False,
            'status': 'running_init_script',
            'message': 'Running parallel initialization...'
        })
    
    # Check if components are initializing
    if initializing:
        return jsonify({
            'is_initialized': False,
            'initializing': True,
            'missing_gemini_key': False,
            'status': 'initializing_components',
            'message': 'Initializing AI components...'
        })
    
    # Check if fully initialized
    if is_initialized:
        return jsonify({
            'is_initialized': True,
            'initializing': False,
            'missing_gemini_key': False,
            'status': 'ready',
            'message': 'System ready'
        })
    
    # Still waiting
    return jsonify({
        'is_initialized': False,
        'initializing': True,
        'missing_gemini_key': False,
        'status': 'waiting',
        'message': 'Waiting for initialization...'
    })

@app.route('/status')
def status():
    """Status endpoint to check initialization progress"""
    return jsonify({
        'is_initialized': is_initialized,
        'initializing': initializing,
        'emotions': galatea_ai.emotional_state if galatea_ai else {'joy': 0.2, 'sadness': 0.2, 'anger': 0.2, 'fear': 0.2, 'curiosity': 0.2},
        'avatar_shape': avatar_engine.avatar_model if avatar_engine and is_initialized else 'Circle',
        'missing_gemini_key': missing_gemini_key
    })

@app.route('/error')
def error_page():
    """Render an informative error page when the app is unavailable"""
    return render_template('error.html', missing_gemini_key=missing_gemini_key)

if __name__ == '__main__':
    print("Starting Galatea Web Interface...")
    print("Initialization will begin automatically when the app starts.")
    
    # Start initialization script immediately when app starts
    logging.info("=" * 70)
    logging.info("STARTING GALATEA AI APPLICATION")
    logging.info("=" * 70)
    logging.info("Launching parallel initialization script...")
    
    # Start initialization script in background thread
    Thread(target=run_init_script, daemon=True).start()
    
    # Start component initialization (will wait for init script)
    Thread(target=initialize_components, daemon=True).start()

    # Add debug logs for avatar shape changes
    logging.info("Avatar system initialized with default shape.")

    # Get port from environment variable (for Hugging Face Spaces compatibility)
    port = int(os.environ.get('PORT', 7860))

    logging.info(f"Flask server starting on port {port}...")
    logging.info("Frontend will poll /api/is_initialized for status")
    
    # Bind to 0.0.0.0 for external access (required for Hugging Face Spaces)
    app.run(host='0.0.0.0', port=port, debug=True)