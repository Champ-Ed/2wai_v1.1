"""
Unified LLM and embeddings client adapter.
Centralizes all OpenAI API calls for consistent usage and easier maintenance.
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from langsmith.wrappers import wrap_openai

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified client for LLM and embeddings operations."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", enable_tracing: bool = True):
        """Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for OpenAI API (allows for custom endpoints)
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.api_key = api_key
        self.base_url = base_url
        self.enable_tracing = enable_tracing
        
        # Create OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Wrap for LangSmith tracing if enabled
        if enable_tracing:
            self.client = wrap_openai(self.client)
            
        logger.debug(f"LLMClient initialized with base_url={base_url}, tracing={enable_tracing}")
    
    def chat(self, messages: List[Dict[str, str]], model: str = "gpt-4o", temperature: float = 0.3, **kwargs) -> Dict[str, Any]:
        """Create a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Chat completion response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def embed(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Create embeddings for the given text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of embedding floats
        """
        if not text or len(text) > 8192:
            logger.warning(f"Invalid text for embedding: length={len(text) if text else 0}")
            return []
            
        try:
            # Use the client's embeddings endpoint for consistency
            response = self.client.embeddings.create(
                model=model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Create embeddings for multiple texts in a single batch.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding lists
        """
        if not texts:
            return []
            
        # Filter out empty or too-long texts
        valid_texts = [text for text in texts if text and len(text) <= 8192]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered {len(texts) - len(valid_texts)} invalid texts from batch")
            
        if not valid_texts:
            return []
            
        try:
            response = self.client.embeddings.create(
                model=model,
                input=valid_texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding creation failed: {e}")
            raise


# Global client instance (will be initialized by the main application)
_global_client: Optional[LLMClient] = None

def get_client() -> LLMClient:
    """Get the global LLM client instance.
    
    Returns:
        The initialized LLM client
        
    Raises:
        RuntimeError: If client hasn't been initialized
    """
    if _global_client is None:
        raise RuntimeError("LLM client not initialized. Call initialize_client() first.")
    return _global_client

def initialize_client(api_key: str, base_url: str = "https://api.openai.com/v1", enable_tracing: bool = True) -> LLMClient:
    """Initialize the global LLM client.
    
    Args:
        api_key: OpenAI API key
        base_url: Base URL for OpenAI API
        enable_tracing: Whether to enable LangSmith tracing
        
    Returns:
        The initialized LLM client
    """
    global _global_client
    _global_client = LLMClient(api_key=api_key, base_url=base_url, enable_tracing=enable_tracing)
    logger.info("Global LLM client initialized")
    return _global_client

def is_initialized() -> bool:
    """Check if the global client has been initialized.
    
    Returns:
        True if client is initialized, False otherwise
    """
    return _global_client is not None
