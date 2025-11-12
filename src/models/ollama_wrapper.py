"""
Ollama Model Wrappers
Provides interfaces for vision and text models with caching and retry logic
"""
import time
import ollama
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OLLAMA_VISION_MODEL, 
    OLLAMA_LLM_MODEL,
    ENABLE_CACHING,
    CACHE_RESPONSES,
    CACHE_DIR,
)
from utils import (
    setup_logging, 
    get_cache_key, 
    cache_result, 
    load_cached_result
)

logger = setup_logging(__name__)


class OllamaModelWrapper:
    """Base wrapper for Ollama models with caching and retry logic"""
    
    def __init__(
        self,
        model_name: str,
        enable_caching: bool = ENABLE_CACHING,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        Initialize Ollama model wrapper
        
        Args:
            model_name: Name of the Ollama model
            enable_caching: Whether to cache responses
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = ollama.Client()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the model is available in Ollama"""
        try:
            # List available models
            models = self.client.list()
            # Use 'model' key (new Ollama API) or 'name' key (old API) as fallback
            model_names = [m.get('model', m.get('name', '')) for m in models.get('models', [])]
            
            # Check if our model is in the list (with or without tag)
            model_available = any(
                self.model_name in name or name.startswith(self.model_name.split(':')[0])
                for name in model_names
            )
            
            if not model_available:
                logger.warning(
                    f"Model {self.model_name} not found in Ollama. "
                    f"Available models: {model_names}\n"
                    f"Please run: ollama pull {self.model_name}"
                )
            else:
                logger.info(f"Model {self.model_name} is available")
        except Exception as e:
            logger.error(f"Failed to verify model availability: {e}")
    
    def _call_with_retry(self, call_func, *args, **kwargs):
        """
        Call a function with retry logic
        
        Args:
            call_func: Function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Function result
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return call_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
        
        raise last_error


class OllamaVisionModel(OllamaModelWrapper):
    """Wrapper for Ollama vision models (e.g., llama3.2-vision)"""
    
    def __init__(self, model_name: str = OLLAMA_VISION_MODEL, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def generate(
        self,
        prompt: str,
        image_path: Union[str, Path],
        temperature: float = 0.0,
        use_cache: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the vision model
        
        Args:
            prompt: Text prompt
            image_path: Path to the image file
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response text
        """
        image_path = str(image_path)
        
        # Check cache
        if use_cache and self.enable_caching:
            cache_key = get_cache_key(
                model=self.model_name,
                prompt=prompt,
                image=image_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cached = load_cached_result(cache_key, CACHE_DIR / "vision")
            if cached:
                logger.debug(f"Cache hit for vision model query")
                return cached['response']
        
        # Make API call
        def _call():
            logger.debug(f"Calling vision model: {self.model_name}")
            
            # Set up options
            options = {'temperature': temperature}
            if max_tokens:
                options['num_predict'] = max_tokens
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path],
                }],
                options=options,
            )
            return response['message']['content']
        
        result = self._call_with_retry(_call)
        
        # Cache result
        if use_cache and self.enable_caching and CACHE_RESPONSES:
            cache_key = get_cache_key(
                model=self.model_name,
                prompt=prompt,
                image=image_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cache_dir = CACHE_DIR / "vision"
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_result(cache_key, {'response': result}, cache_dir)
        
        return result
    
    def generate_with_context(
        self,
        prompt: str,
        image_path: Union[str, Path],
        context: Optional[str] = None,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> str:
        """
        Generate a response with additional context
        
        Args:
            prompt: Main prompt
            image_path: Path to the image
            context: Additional context to include
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            
        Returns:
            Model response text
        """
        if context:
            full_prompt = f"{context}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        return self.generate(full_prompt, image_path, temperature, use_cache)


class OllamaTextModel(OllamaModelWrapper):
    """Wrapper for Ollama text-only models (e.g., qwen3:8b)"""
    
    def __init__(self, model_name: str = OLLAMA_LLM_MODEL, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        use_cache: bool = True,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the text model
        
        Args:
            prompt: Text prompt
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            system_prompt: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response text
        """
        # Check cache
        if use_cache and self.enable_caching:
            cache_key = get_cache_key(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                system=system_prompt,
                max_tokens=max_tokens,
            )
            cached = load_cached_result(cache_key, CACHE_DIR / "text")
            if cached:
                logger.debug(f"Cache hit for text model query")
                return cached['response']
        
        # Make API call
        def _call():
            logger.debug(f"Calling text model: {self.model_name}")
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            # Set up options
            options = {'temperature': temperature}
            if max_tokens:
                options['num_predict'] = max_tokens
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )
            return response['message']['content']
        
        result = self._call_with_retry(_call)
        
        # Cache result
        if use_cache and self.enable_caching and CACHE_RESPONSES:
            cache_key = get_cache_key(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                system=system_prompt,
                max_tokens=max_tokens,
            )
            cache_dir = CACHE_DIR / "text"
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_result(cache_key, {'response': result}, cache_dir)
        
        return result


def test_models():
    """Test the model wrappers"""
    logger.info("Testing Ollama Model Wrappers")
    
    # Test text model
    logger.info("\n=== Testing Text Model ===")
    text_model = OllamaTextModel()
    response = text_model.generate(
        "What is 2+2? Answer with just the number.",
        temperature=0.0
    )
    logger.info(f"Text model response: {response}")
    
    # Test vision model (if we have an image)
    logger.info("\n=== Testing Vision Model ===")
    # We'll test this in the main pipeline
    logger.info("Vision model test requires an image - will test in pipeline")
    
    logger.info("\nModel wrapper tests completed!")


if __name__ == "__main__":
    test_models()
