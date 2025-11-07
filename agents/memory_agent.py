"""Memory Agent - responsible for memory retrieval and storage"""
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

class MemoryAgent:
    """Agent responsible for memory retrieval and storage"""
    
    def __init__(self, memory_system, config=None):
        self.memory_system = memory_system
        self.config = config or MODEL_CONFIG or {}
    
    def retrieve_memories(self, query, n_results=None):
        """Retrieve relevant memories for a query"""
        if n_results is None:
            max_memories = self.config.get('memory', {}).get('retrieval', {}).get('max_retrieved_memories', 5) if self.config else 5
        else:
            max_memories = n_results
        
        try:
            memories = self.memory_system.retrieve_relevant_memories(query, n_results=max_memories)
            if memories:
                logging.info(f"[MemoryAgent] Retrieved {len(memories)} relevant memories")
            return memories
        except Exception as e:
            logging.error(f"[MemoryAgent] Error retrieving memories: {e}")
            return []
    
    def store_memory(self, text, metadata=None, memory_type="conversation"):
        """Store a memory"""
        try:
            self.memory_system.store_memory(text, metadata, memory_type)
            logging.info(f"[MemoryAgent] Stored memory: {memory_type}")
        except Exception as e:
            logging.error(f"[MemoryAgent] Error storing memory: {e}")
    
    def is_ready(self):
        """Check if memory agent is ready"""
        return self.memory_system.is_ready() if self.memory_system else False

