"""Memory system using JSON for simple key-value storage"""
import os
import json
import logging
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

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

