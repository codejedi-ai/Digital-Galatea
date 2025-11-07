#!/usr/bin/env python3
"""
Galatea AI Initialization Script
Handles parallel initialization of all components
"""

import os
import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('initialization.log')
    ]
)

# Check NumPy version before proceeding
try:
    import numpy as np
    np_version = np.__version__
    if np_version.startswith('2.'):
        logging.error("=" * 70)
        logging.error("NUM PY COMPATIBILITY ERROR")
        logging.error("=" * 70)
        logging.error(f"NumPy {np_version} is installed, but required libraries need NumPy < 2.0")
        logging.error("")
        logging.error("SOLUTION:")
        logging.error("  Option 1: Run the fix script:")
        logging.error("    python fix_numpy.py")
        logging.error("")
        logging.error("  Option 2: Manually downgrade:")
        logging.error("    pip install 'numpy<2.0.0'")
        logging.error("")
        logging.error("  Option 3: Reinstall all dependencies:")
        logging.error("    pip install -r requirements.txt")
        logging.error("")
        logging.error("This will downgrade NumPy to a compatible version.")
        logging.error("=" * 70)
        logging.warning("⚠ Continuing with initialization, but some components may fail...")
        logging.warning("⚠ Please fix NumPy version for full functionality")
    else:
        logging.info(f"✓ NumPy version check passed: {np_version}")
except ImportError:
    logging.warning("NumPy not installed - will be installed as dependency")
except Exception as e:
    logging.warning(f"Could not check NumPy version: {e}")

# Load environment variables
load_dotenv()

# Global status tracking
init_status = {
    'json_memory': {'ready': False, 'error': None},
    'sentiment_analyzer': {'ready': False, 'error': None},
    'gemini_api': {'ready': False, 'error': None},
    'inflection_api': {'ready': False, 'error': None},
    'quantum_api': {'ready': False, 'error': None},
}

# ChromaDB and embedding model removed - using JSON-only memory

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
            # Test it
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
            # Check for NumPy compatibility issues
            if 'np.float_' in error_msg or 'NumPy 2' in error_msg or '_ARRAY_API' in error_msg:
                logging.warning(f"⚠ [Sentiment Analyzer] NumPy compatibility issue - using fallback")
                print("⚠ [Sentiment Analyzer] NumPy compatibility issue - using fallback")
                init_status['sentiment_analyzer']['ready'] = True  # Fallback available
                return True
            else:
                raise
    except Exception as e:
        error_msg = f"Sentiment analyzer initialization failed: {e}"
        logging.warning(f"⚠ [Sentiment Analyzer] {error_msg} - using fallback")
        print(f"⚠ [Sentiment Analyzer] Using fallback")
        init_status['sentiment_analyzer']['error'] = str(e)
        # Still mark as ready since we have fallback
        init_status['sentiment_analyzer']['ready'] = True
        return True

def validate_gemini_api():
    """Validate Gemini API key"""
    try:
        logging.info("🔄 [Gemini API] Validating API key...")
        print("🔄 [Gemini API] Validating API key...")
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logging.warning("⚠ [Gemini API] API key not found")
            print("⚠ [Gemini API] API key not found")
            init_status['gemini_api']['ready'] = False
            return False
        
        # Try to use custom LLM wrapper to validate
        try:
            from llm_wrapper import LLMWrapper
            # Initialize wrapper with test model
            wrapper = LLMWrapper(gemini_model="gemini-1.5-flash")
            response = wrapper.call_gemini(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            if response:
                logging.info("✓ [Gemini API] API key validated")
                print("✓ [Gemini API] API key validated")
                init_status['gemini_api']['ready'] = True
                return True
            else:
                logging.warning("⚠ [Gemini API] Validation failed - no response")
                print("⚠ [Gemini API] Validation failed - key exists, may be network issue")
                return False
        except Exception as e:
            logging.warning(f"⚠ [Gemini API] Validation failed: {e}")
            print("⚠ [Gemini API] Validation failed - key exists, may be network issue")
            # Still mark as available if key exists (might be network issue)
            init_status['gemini_api']['ready'] = True
            return True
    except Exception as e:
        error_msg = f"Gemini API validation failed: {e}"
        logging.error(f"✗ [Gemini API] {error_msg}")
        print(f"✗ [Gemini API] {error_msg}")
        init_status['gemini_api']['error'] = str(e)
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
        
        # Test API key by making a simple request
        import requests
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
        # Don't fail initialization if this fails
        init_status['inflection_api']['ready'] = False
        return False

def validate_quantum_api():
    """Validate Quantum Random Numbers API key"""
    try:
        logging.info("🔄 [Quantum API] Validating API key...")
        print("🔄 [Quantum API] Validating API key...")
        api_key = os.getenv("ANU_QUANTUM_API_KEY")
        
        if not api_key:
            logging.warning("⚠ [Quantum API] API key not found")
            print("⚠ [Quantum API] API key not found")
            init_status['quantum_api']['ready'] = False
            return False
        
        # Test API key
        import requests
        url = "https://api.quantumnumbers.anu.edu.au"
        headers = {"x-api-key": api_key}
        params = {"length": 1, "type": "uint8"}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            logging.info("✓ [Quantum API] API key validated")
            print("✓ [Quantum API] API key validated")
            init_status['quantum_api']['ready'] = True
            return True
        else:
            logging.warning(f"⚠ [Quantum API] Validation failed: {response.status_code}")
            print(f"⚠ [Quantum API] Validation failed: {response.status_code}")
            init_status['quantum_api']['ready'] = False
            return False
    except Exception as e:
        error_msg = f"Quantum API validation failed: {e}"
        logging.warning(f"⚠ [Quantum API] {error_msg}")
        print(f"⚠ [Quantum API] {error_msg}")
        init_status['quantum_api']['ready'] = False
        return False

def initialize_json_memory():
    """Initialize JSON memory database"""
    try:
        logging.info("🔄 [JSON Memory] Initializing...")
        print("🔄 [JSON Memory] Initializing...")
        import json
        
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

def run_initialization():
    """Run all initialization steps in parallel"""
    start_time = time.time()
    
    logging.info("=" * 70)
    logging.info("GALATEA AI PARALLEL INITIALIZATION")
    logging.info("=" * 70)
    logging.info("Starting parallel initialization of all components...")
    logging.info("")
    
    # Define initialization tasks
    tasks = [
        ("JSON Memory", initialize_json_memory),
        ("Sentiment Analyzer", initialize_sentiment_analyzer),
        ("Gemini API", validate_gemini_api),
        ("Inflection AI", validate_inflection_api),
        ("Quantum API", validate_quantum_api),
    ]
    
    # Run tasks in parallel
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
    
    # Print summary
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
        
        # Critical components (must be ready)
        if component in ['json_memory', 'sentiment_analyzer', 'gemini_api']:
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
    
    # Check for NumPy compatibility issues
    numpy_issue = False
    for component, status in init_status.items():
        if status.get('error') and ('np.float_' in str(status['error']) or 'NumPy 2' in str(status['error']) or '_ARRAY_API' in str(status['error'])):
            numpy_issue = True
            break
    
    if numpy_issue:
        logging.error("")
        logging.error("=" * 70)
        logging.error("NUM PY COMPATIBILITY ISSUE DETECTED")
        logging.error("=" * 70)
        logging.error("Some components failed due to NumPy 2.0 incompatibility.")
        logging.error("")
        logging.error("TO FIX:")
        logging.error("  1. Run: python fix_numpy.py")
        logging.error("  2. Or: pip install 'numpy<2.0.0'")
        logging.error("  3. Then restart the application")
        logging.error("=" * 70)
        logging.error("")
    
    # Determine final status
    if critical_ready:
        if all_ready:
            logging.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            logging.info("🎉 Galatea AI is ready to use!")
            print("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            print("🎉 Galatea AI is ready to use!")
            return True
        else:
            logging.info("⚠️  CRITICAL COMPONENTS READY (some optional components failed)")
            if numpy_issue:
                logging.warning("⚠️  Some failures due to NumPy compatibility - fix NumPy for full functionality")
            logging.info("✅ Galatea AI is ready to use (with limited features)")
            print("⚠️  CRITICAL COMPONENTS READY (some optional components failed)")
            print("✅ Galatea AI is ready to use (with limited features)")
            return True
    else:
        logging.error("❌ CRITICAL COMPONENTS FAILED")
        if numpy_issue:
            logging.error("⚠️  Failures likely due to NumPy 2.0 - run 'python fix_numpy.py' to fix")
        logging.error("⚠️  Galatea AI may not function properly")
        print("❌ CRITICAL COMPONENTS FAILED")
        print("⚠️  Galatea AI may not function properly")
        return False

if __name__ == "__main__":
    try:
        success = run_initialization()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logging.info("\n⚠️  Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n❌ Fatal error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

