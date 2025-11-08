from flask import Flask, render_template, request, jsonify, url_for
import os
import sys
import time
import json
from dotenv import load_dotenv
import logging
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import nltk
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Debug: Log environment variable status
logging.info("=" * 60)
logging.info("ENVIRONMENT VARIABLES CHECK")
logging.info("=" * 60)
deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
missing_deepseek_key = False
if deepseek_key:
    logging.info(f"✓ DEEPSEEK_API_KEY found (length: {len(deepseek_key)} chars)")
    logging.info(f"  First 10 chars: {deepseek_key[:10]}...")
else:
    missing_deepseek_key = True
    logging.error("=" * 60)
    logging.error("✗ DEEPSEEK_API_KEY not found in environment!")
    logging.error("=" * 60)
    logging.error("")
    logging.error("The DEEPSEEK_API_KEY environment variable is required for full functionality.")
    logging.error("")
    logging.error("For Hugging Face Spaces:")
    logging.error("  1. Go to Settings → Repository secrets")
    logging.error("  2. Click 'New secret'")
    logging.error("  3. Name: DEEPSEEK_API_KEY")
    logging.error("  4. Value: [Your DeepSeek API key]")
    logging.error("  5. Get a key from: https://platform.deepseek.com/")
    logging.error("")
    logging.error("For local development:")
    logging.error("  1. Copy .env.example to .env")
    logging.error("  2. Add your API key to the .env file")
    logging.error("")
    logging.error("Available env vars starting with 'DEEPSEEK': " +
                 str([k for k in os.environ.keys() if 'DEEPSEEK' in k.upper()]))
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
deepseek_initialized = False
max_init_retries = 3
current_init_retry = 0

# Quantum numbers queue for /api/avatar endpoint
quantum_numbers_queue = deque(maxlen=100)
quantum_queue_lock = Lock()
quantum_filling = False

# Check for required environment variables
required_env_vars = ['DEEPSEEK_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logging.error("Please set these in your .env file or environment")
    print(f"⚠️ Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these in your .env file or environment")

def initialize_deepseek():
    """Initialize DeepSeek API specifically"""
    global deepseek_initialized
    
    if not galatea_ai:
        logging.warning("Cannot initialize DeepSeek: GalateaAI instance not created yet")
        return False
        
    if missing_deepseek_key:
        logging.error("Cannot initialize DeepSeek: DEEPSEEK_API_KEY is missing")
        return False

    try:
        # Check for DEEPSEEK_API_KEY
        if not os.environ.get('DEEPSEEK_API_KEY'):
            logging.error("DEEPSEEK_API_KEY not found in environment variables")
            return False
            
        # Check if DeepSeek agent is ready (initialization happens automatically in GalateaAI.__init__)
        deepseek_success = hasattr(galatea_ai, 'deepseek_agent') and galatea_ai.deepseek_agent.is_ready()
        
        if deepseek_success:
            deepseek_initialized = True
            logging.info("DeepSeek API initialized successfully")
            return True
        else:
            logging.error("Failed to initialize DeepSeek API")
            return False
    except Exception as e:
        logging.error(f"Error initializing DeepSeek API: {e}")
        return False

# Global status tracking for parallel initialization
init_status = {
    'json_memory': {'ready': False, 'error': None},
    'sentiment_analyzer': {'ready': False, 'error': None},
    'deepseek_api': {'ready': False, 'error': None},
    'inflection_api': {'ready': False, 'error': None},
    'quantum_api': {'ready': False, 'error': None},
}

def initialize_json_memory():
    """Initialize JSON memory database"""
    try:
        logging.info("🔄 [JSON Memory] Initializing...")
        print("🔄 [JSON Memory] Initializing...")
        json_path = "./memory.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            logging.info(f"✓ [JSON Memory] Loaded {len(memory)} entries")
            print(f"✓ [JSON Memory] Loaded {len(memory)} entries")
        else:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logging.info("✓ [JSON Memory] Created new database")
            print("✓ [JSON Memory] Created new database")
        init_status['json_memory']['ready'] = True
        return True
    except Exception as e:
        error_msg = f"JSON memory initialization failed: {e}"
        logging.error(f"✗ [JSON Memory] {error_msg}")
        print(f"✗ [JSON Memory] {error_msg}")
        init_status['json_memory']['error'] = str(e)
        return False

def initialize_sentiment_analyzer():
    """Initialize sentiment analyzer"""
    try:
        logging.info("🔄 [Sentiment Analyzer] Starting initialization...")
        print("🔄 [Sentiment Analyzer] Starting initialization...")
        try:
            from transformers import pipeline
            analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            )
            result = analyzer("test")
            logging.info("✓ [Sentiment Analyzer] Hugging Face model loaded")
            print("✓ [Sentiment Analyzer] Hugging Face model loaded")
            init_status['sentiment_analyzer']['ready'] = True
            return True
        except ImportError:
            logging.info("✓ [Sentiment Analyzer] Using fallback (NLTK VADER)")
            print("✓ [Sentiment Analyzer] Using fallback (NLTK VADER)")
            init_status['sentiment_analyzer']['ready'] = True
            return True
        except Exception as e:
            error_msg = str(e)
            if 'np.float_' in error_msg or 'NumPy 2' in error_msg or '_ARRAY_API' in error_msg:
                logging.warning(f"⚠ [Sentiment Analyzer] NumPy compatibility issue - using fallback")
                print("⚠ [Sentiment Analyzer] NumPy compatibility issue - using fallback")
                init_status['sentiment_analyzer']['ready'] = True
                return True
            else:
                raise
    except Exception as e:
        error_msg = f"Sentiment analyzer initialization failed: {e}"
        logging.warning(f"⚠ [Sentiment Analyzer] {error_msg} - using fallback")
        print(f"⚠ [Sentiment Analyzer] Using fallback")
        init_status['sentiment_analyzer']['error'] = str(e)
        init_status['sentiment_analyzer']['ready'] = True
        return True

def validate_deepseek_api():
    """Validate DeepSeek API key"""
    try:
        logging.info("🔄 [DeepSeek API] Validating API key...")
        print("🔄 [DeepSeek API] Validating API key...")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logging.warning("⚠ [DeepSeek API] API key not found")
            print("⚠ [DeepSeek API] API key not found")
            init_status['deepseek_api']['ready'] = False
            return False
        try:
            from llm_wrapper import LLMWrapper
            from config import MODEL_CONFIG
            
            # Get model from config
            deepseek_config = MODEL_CONFIG.get('deepseek', {}) if MODEL_CONFIG else {}
            deepseek_model = deepseek_config.get('model', 'deepseek-reasoner')
            
            wrapper = LLMWrapper(deepseek_model=deepseek_model)
            response = wrapper.call_deepseek(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            if response:
                logging.info("✓ [DeepSeek API] API key validated")
                print("✓ [DeepSeek API] API key validated")
                init_status['deepseek_api']['ready'] = True
                return True
            else:
                logging.warning("⚠ [DeepSeek API] Validation failed - no response")
                print("⚠ [DeepSeek API] Validation failed - key exists, may be network issue")
                return False
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            # Check status code from exception if available
            status_code = getattr(e, 'status_code', None)
            response_text = getattr(e, 'response_text', error_msg)
            
            # Log the full error for debugging
            logging.error(f"✗ [DeepSeek API] Validation exception: {error_type}: {error_msg}")
            print(f"✗ [DeepSeek API] Validation exception: {error_type}: {error_msg}")
            
            # Check if it's a 404 (model not found) - this is a real error
            if status_code == 404 or '404' in error_msg or 'NOT_FOUND' in error_msg:
                logging.error(f"✗ [DeepSeek API] Model not found: {error_msg}")
                print(f"✗ [DeepSeek API] Model not found - check models.yaml configuration")
                init_status['deepseek_api']['error'] = error_msg
                return False
            # Check if it's a 429 (rate limit/quota exceeded) - API key is valid, just quota issue
            elif status_code == 429 or '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in response_text.lower():
                logging.info("ℹ️  [DeepSeek API] Rate limit/quota exceeded (API key is valid)")
                print("ℹ️  [DeepSeek API] Rate limit/quota exceeded (API key is valid, will work when quota resets)")
                init_status['deepseek_api']['ready'] = True  # Key is valid, just quota issue
                init_status['deepseek_api']['error'] = "Rate limit/quota exceeded"
                return True  # Don't fail initialization - key is valid
            # Check if it's an authentication error (401) - invalid API key
            elif status_code == 401 or '401' in error_msg or 'unauthorized' in error_msg.lower() or 'authentication' in error_msg.lower():
                logging.error(f"✗ [DeepSeek API] Authentication failed - invalid API key: {error_msg}")
                print(f"✗ [DeepSeek API] Authentication failed - check your DEEPSEEK_API_KEY")
                init_status['deepseek_api']['error'] = f"Authentication failed: {error_msg}"
                return False
            else:
                logging.warning(f"⚠ [DeepSeek API] Validation failed: {e}")
                print(f"⚠ [DeepSeek API] Validation failed - key exists, may be network issue. Error: {error_msg}")
                init_status['deepseek_api']['ready'] = False
                init_status['deepseek_api']['error'] = error_msg
                return False
    except Exception as e:
        error_msg = f"DeepSeek API validation failed: {e}"
        logging.error(f"✗ [DeepSeek API] {error_msg}")
        print(f"✗ [DeepSeek API] {error_msg}")
        init_status['deepseek_api']['error'] = str(e)
        return False

def validate_inflection_api():
    """Validate Inflection AI API key"""
    try:
        logging.info("🔄 [Inflection AI] Validating API key...")
        print("🔄 [Inflection AI] Validating API key...")
        api_key = os.getenv("INFLECTION_AI_API_KEY")
        if not api_key:
            logging.warning("⚠ [Inflection AI] API key not found")
            print("⚠ [Inflection AI] API key not found")
            init_status['inflection_api']['ready'] = False
            return False
        url = "https://api.inflection.ai/external/api/inference"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "context": [{"text": "test", "type": "Human"}],
            "config": "Pi-3.1"
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            logging.info("✓ [Inflection AI] API key validated")
            print("✓ [Inflection AI] API key validated")
            init_status['inflection_api']['ready'] = True
            return True
        else:
            logging.warning(f"⚠ [Inflection AI] Validation failed: {response.status_code}")
            print(f"⚠ [Inflection AI] Validation failed: {response.status_code}")
            init_status['inflection_api']['ready'] = False
            return False
    except Exception as e:
        error_msg = f"Inflection AI validation failed: {e}"
        logging.warning(f"⚠ [Inflection AI] {error_msg}")
        print(f"⚠ [Inflection AI] {error_msg}")
        init_status['inflection_api']['ready'] = False
        return False

def validate_quantum_api():
    """Validate Quantum Random Numbers API key (optional component)"""
    try:
        logging.info("🔄 [Quantum API] Validating API key...")
        print("🔄 [Quantum API] Validating API key...")
        api_key = os.getenv("ANU_QUANTUM_API_KEY")
        if not api_key:
            logging.info("ℹ️  [Quantum API] API key not found (optional - will use pseudo-random)")
            print("ℹ️  [Quantum API] API key not found (optional - will use pseudo-random)")
            init_status['quantum_api']['ready'] = False
            return True  # Not an error - optional component
        url = "https://api.quantumnumbers.anu.edu.au"
        headers = {"x-api-key": api_key}
        params = {"length": 1, "type": "uint8"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            logging.info("✓ [Quantum API] API key validated")
            print("✓ [Quantum API] API key validated")
            init_status['quantum_api']['ready'] = True
            return True
        elif response.status_code == 429:
            # Rate limit - not an error, just unavailable temporarily
            logging.info("ℹ️  [Quantum API] Rate limited (optional - will use pseudo-random)")
            print("ℹ️  [Quantum API] Rate limited (optional - will use pseudo-random)")
            init_status['quantum_api']['ready'] = False
            return True  # Not an error - optional component
        else:
            logging.info(f"ℹ️  [Quantum API] Validation failed: {response.status_code} (optional - will use pseudo-random)")
            print(f"ℹ️  [Quantum API] Validation failed: {response.status_code} (optional - will use pseudo-random)")
            init_status['quantum_api']['ready'] = False
            return True  # Not an error - optional component
    except Exception as e:
        # Any exception is not critical - quantum randomness is optional
        logging.info(f"ℹ️  [Quantum API] Unavailable: {e} (optional - will use pseudo-random)")
        print(f"ℹ️  [Quantum API] Unavailable: {e} (optional - will use pseudo-random)")
        init_status['quantum_api']['ready'] = False
        return True  # Not an error - optional component

def run_parallel_initialization():
    """Run all initialization steps in parallel"""
    start_time = time.time()
    
    logging.info("=" * 70)
    logging.info("GALATEA AI PARALLEL INITIALIZATION")
    logging.info("=" * 70)
    logging.info("Starting parallel initialization of all components...")
    logging.info("")
    print("=" * 70)
    print("GALATEA AI PARALLEL INITIALIZATION")
    print("=" * 70)
    print("Starting parallel initialization of all components...")
    print("")
    
    tasks = [
        ("JSON Memory", initialize_json_memory),
        ("Sentiment Analyzer", initialize_sentiment_analyzer),
        ("DeepSeek API", validate_deepseek_api),
        ("Inflection AI", validate_inflection_api),
        ("Quantum API", validate_quantum_api),
    ]
    
    completed_count = 0
    total_tasks = len(tasks)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(task[1]): task[0] for task in tasks}
        
        for future in as_completed(futures):
            task_name = futures[future]
            completed_count += 1
            try:
                result = future.result()
                if result:
                    logging.info(f"✅ [{task_name}] Completed successfully ({completed_count}/{total_tasks})")
                    print(f"✅ [{task_name}] Completed successfully ({completed_count}/{total_tasks})")
                else:
                    logging.warning(f"⚠️  [{task_name}] Completed with warnings ({completed_count}/{total_tasks})")
                    print(f"⚠️  [{task_name}] Completed with warnings ({completed_count}/{total_tasks})")
            except Exception as e:
                logging.error(f"❌ [{task_name}] Failed: {e} ({completed_count}/{total_tasks})")
                print(f"❌ [{task_name}] Failed: {e} ({completed_count}/{total_tasks})")
    
    elapsed_time = time.time() - start_time
    
    logging.info("")
    logging.info("=" * 70)
    logging.info("INITIALIZATION SUMMARY")
    logging.info("=" * 70)
    print("")
    print("=" * 70)
    print("INITIALIZATION SUMMARY")
    print("=" * 70)
    
    all_ready = True
    critical_ready = True
    
    for component, status in init_status.items():
        status_icon = "✓" if status['ready'] else "✗"
        error_info = f" - {status['error']}" if status['error'] else ""
        status_msg = f"{status_icon} {component.upper()}: {'READY' if status['ready'] else 'FAILED'}{error_info}"
        logging.info(status_msg)
        print(status_msg)
        
        if component in ['json_memory', 'sentiment_analyzer', 'deepseek_api']:
            if not status['ready']:
                critical_ready = False
        
        if not status['ready']:
            all_ready = False
    
    logging.info("")
    logging.info(f"⏱️  Total initialization time: {elapsed_time:.2f} seconds")
    logging.info("")
    print("")
    print(f"⏱️  Total initialization time: {elapsed_time:.2f} seconds")
    print("")
    
    if critical_ready:
        if all_ready:
            logging.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            logging.info("🎉 Galatea AI is ready to use!")
            print("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            print("🎉 Galatea AI is ready to use!")
            return True
        else:
            logging.info("⚠️  CRITICAL COMPONENTS READY (some optional components failed)")
            logging.info("✅ Galatea AI is ready to use (with limited features)")
            print("⚠️  CRITICAL COMPONENTS READY (some optional components failed)")
            print("✅ Galatea AI is ready to use (with limited features)")
            return True
    else:
        logging.error("❌ CRITICAL COMPONENTS FAILED")
        logging.error("⚠️  Galatea AI may not function properly")
        print("❌ CRITICAL COMPONENTS FAILED")
        print("⚠️  Galatea AI may not function properly")
        return False

def initialize_components():
    """Initialize Galatea components"""
    global galatea_ai, dialogue_engine, avatar_engine, is_initialized, initializing
    global current_init_retry, deepseek_initialized
    
    if initializing or is_initialized:
        return

    if missing_deepseek_key:
        logging.error("Initialization aborted: DEEPSEEK_API_KEY missing")
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
        logging.info(f"  - DeepSeek available: {init_status['deepseek_available']}")
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
        # Start component initialization if not already started
        if not is_initialized and not initializing and not missing_deepseek_key:
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
    if missing_deepseek_key:
        return jsonify({
            'error': 'DEEPSEEK_API_KEY is missing. Chat is unavailable.',
            'status': 'missing_deepseek_key',
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

def fill_quantum_queue():
    """Asynchronously fill the quantum numbers queue with 100 numbers"""
    global quantum_filling, quantum_numbers_queue
    
    with quantum_queue_lock:
        if quantum_filling:
            return  # Already filling
        quantum_filling = True
    
    def _fill_queue():
        global quantum_filling
        try:
            quantum_api_key = os.getenv("ANU_QUANTUM_API_KEY")
            if not quantum_api_key:
                logging.debug("[Quantum Queue] No API key, using pseudo-random")
                import random
                with quantum_queue_lock:
                    while len(quantum_numbers_queue) < 100:
                        quantum_numbers_queue.append(random.random())
                return
            
            from config import MODEL_CONFIG
            quantum_config = MODEL_CONFIG.get('quantum', {}) if MODEL_CONFIG else {}
            api_endpoint = quantum_config.get('api_endpoint', 'https://api.quantumnumbers.anu.edu.au')
            
            headers = {"x-api-key": quantum_api_key}
            params = {"length": 1, "type": "uint8"}
            
            numbers_fetched = 0
            with quantum_queue_lock:
                current_size = len(quantum_numbers_queue)
            
            while numbers_fetched < 100:
                try:
                    response = requests.get(api_endpoint, headers=headers, params=params, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success') and 'data' in result and len(result['data']) > 0:
                            normalized = result['data'][0] / 255.0
                            with quantum_queue_lock:
                                quantum_numbers_queue.append(normalized)
                            numbers_fetched += 1
                        else:
                            # Fallback to pseudo-random
                            import random
                            with quantum_queue_lock:
                                quantum_numbers_queue.append(random.random())
                            numbers_fetched += 1
                    elif response.status_code == 429:
                        # Rate limited - use pseudo-random
                        import random
                        with quantum_queue_lock:
                            quantum_numbers_queue.append(random.random())
                        numbers_fetched += 1
                    else:
                        # Error - use pseudo-random
                        import random
                        with quantum_queue_lock:
                            quantum_numbers_queue.append(random.random())
                        numbers_fetched += 1
                except Exception as e:
                    logging.debug(f"[Quantum Queue] Error fetching number: {e}, using pseudo-random")
                    import random
                    with quantum_queue_lock:
                        quantum_numbers_queue.append(random.random())
                    numbers_fetched += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
            
            logging.info(f"[Quantum Queue] Filled queue with {numbers_fetched} numbers")
        except Exception as e:
            logging.error(f"[Quantum Queue] Error filling queue: {e}")
        finally:
            with quantum_queue_lock:
                quantum_filling = False
    
    # Start filling in background thread
    Thread(target=_fill_queue, daemon=True).start()

def pop_quantum_number():
    """Pop a quantum number from the queue, or return pseudo-random if empty"""
    global quantum_numbers_queue
    
    with quantum_queue_lock:
        if len(quantum_numbers_queue) > 0:
            return quantum_numbers_queue.popleft()
        else:
            # Queue is empty, use pseudo-random
            import random
            return random.random()

@app.route('/api/avatar')
def get_avatar():
    """Endpoint to get the current avatar shape and state with enhanced responsiveness"""
    global last_avatar_update, quantum_numbers_queue, quantum_filling
    
    if not is_initialized:
        return jsonify({
            'avatar_shape': 'Circle',
            'is_initialized': False,
            'last_updated': last_avatar_update,
            'status': 'initializing'
        })
    
    try:
        # Check if quantum queue is empty, fill it asynchronously if needed
        with quantum_queue_lock:
            queue_empty = len(quantum_numbers_queue) == 0
        
        if queue_empty and not quantum_filling:
            fill_quantum_queue()
        
        # Pop one quantum number to update emotions
        quantum_num = pop_quantum_number()
        
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
            # Apply quantum influence to emotions
            emotions = ["joy", "sadness", "anger", "fear", "curiosity"]
            # Use quantum number to influence a random emotion
            import random
            emotion_index = int(quantum_num * len(emotions)) % len(emotions)
            selected_emotion = emotions[emotion_index]
            
            # Apply subtle quantum influence (-0.05 to +0.05)
            influence = (quantum_num - 0.5) * 0.1
            current_value = galatea_ai.emotional_state[selected_emotion]
            new_value = max(0.05, min(1.0, current_value + influence))
            galatea_ai.emotional_state[selected_emotion] = new_value
            
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
            
            # Save emotional state to JSON
            if hasattr(galatea_ai, 'emotional_agent'):
                galatea_ai.emotional_agent._save_to_json()
            
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
        'deepseek_available': hasattr(galatea_ai, 'deepseek_available') and galatea_ai.deepseek_available if galatea_ai else False,
        'is_initialized': is_initialized,
        'missing_deepseek_key': missing_deepseek_key
    })

@app.route('/api/availability')
def availability():
    """Report overall availability state to the frontend"""
    if missing_deepseek_key:
        return jsonify({
            'available': False,
            'status': 'missing_deepseek_key',
            'is_initialized': False,
            'initializing': False,
            'missing_deepseek_key': True,
            'error_page': url_for('error_page')
        })

    if initializing or not is_initialized:
        return jsonify({
            'available': False,
            'status': 'initializing',
            'is_initialized': is_initialized,
            'initializing': initializing,
            'missing_deepseek_key': False
        })

    return jsonify({
        'available': True,
        'status': 'ready',
        'is_initialized': True,
        'initializing': False,
        'missing_deepseek_key': False
    })

@app.route('/api/is_initialized')
def is_initialized_endpoint():
    """Lightweight endpoint for polling initialization progress"""
    # Determine current initialization state
    if missing_deepseek_key:
        return jsonify({
            'is_initialized': False,
            'initializing': False,
            'missing_deepseek_key': True,
            'error_page': url_for('error_page'),
            'status': 'missing_api_key'
        })
    
    # Check if components are initializing
    if initializing:
        return jsonify({
            'is_initialized': False,
            'initializing': True,
            'missing_deepseek_key': False,
            'status': 'initializing_components',
            'message': 'Initializing AI components...'
        })
    
    # Check if fully initialized
    if is_initialized:
        return jsonify({
            'is_initialized': True,
            'initializing': False,
            'missing_deepseek_key': False,
            'status': 'ready',
            'message': 'System ready'
        })
    
    # Still waiting
    return jsonify({
        'is_initialized': False,
        'initializing': True,
        'missing_deepseek_key': False,
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
        'missing_deepseek_key': missing_deepseek_key
    })

@app.route('/error')
def error_page():
    """Render an informative error page when the app is unavailable"""
    return render_template('error.html', missing_deepseek_key=missing_deepseek_key)

if __name__ == '__main__':
    print("Starting Galatea Web Interface...")
    
    # Run parallel initialization BEFORE starting Flask app
    logging.info("=" * 70)
    logging.info("STARTING GALATEA AI APPLICATION")
    logging.info("=" * 70)
    logging.info("Running parallel initialization...")
    print("=" * 70)
    print("STARTING GALATEA AI APPLICATION")
    print("=" * 70)
    print("Running parallel initialization...")
    print("")
    
    # Run parallel initialization synchronously
    init_success = run_parallel_initialization()
    
    if not init_success:
        logging.error("=" * 70)
        logging.error("CRITICAL: Parallel initialization failed")
        logging.error("Application will exit")
        logging.error("=" * 70)
        print("=" * 70)
        print("CRITICAL: Parallel initialization failed")
        print("Application will exit")
        print("=" * 70)
        sys.exit(1)
    
    # Now initialize Galatea components
    logging.info("Initializing Galatea AI components...")
    print("Initializing Galatea AI components...")
    initialize_components()
    
    if not is_initialized:
        logging.error("=" * 70)
        logging.error("CRITICAL: Component initialization failed")
        logging.error("Application will exit")
        logging.error("=" * 70)
        print("=" * 70)
        print("CRITICAL: Component initialization failed")
        print("Application will exit")
        print("=" * 70)
        sys.exit(1)

    # Add debug logs for avatar shape changes
    logging.info("Avatar system initialized with default shape.")

    # Get port from environment variable (for Hugging Face Spaces compatibility)
    port = int(os.environ.get('PORT', 7860))

    logging.info(f"Flask server starting on port {port}...")
    logging.info("Frontend will poll /api/is_initialized for status")
    print(f"\nFlask server starting on port {port}...")
    print("Frontend will poll /api/is_initialized for status\n")

    # Bind to 0.0.0.0 for external access (required for Hugging Face Spaces)
    app.run(host='0.0.0.0', port=port, debug=True)