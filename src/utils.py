"""
Utility functions for ARE-VQA Pipeline
"""
import json
import logging
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys

from config import LOGS_DIR, CACHE_DIR, LOG_LEVEL


def setup_logging(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_json(data: Any, filepath: Path, indent: int = 2):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        MD5 hash of the arguments
    """
    # Convert args and kwargs to a stable string representation
    cache_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(cache_str.encode()).hexdigest()


def cache_result(cache_key: str, result: Any, cache_dir: Path = CACHE_DIR):
    """
    Cache a result to disk
    
    Args:
        cache_key: Unique key for the cache
        result: Result to cache
        cache_dir: Directory to store cache
    """
    cache_file = cache_dir / f"{cache_key}.json"
    save_json(result, cache_file)


def load_cached_result(cache_key: str, cache_dir: Path = CACHE_DIR) -> Optional[Any]:
    """
    Load a cached result from disk
    
    Args:
        cache_key: Unique key for the cache
        cache_dir: Directory to load cache from
        
    Returns:
        Cached result or None if not found
    """
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        return load_json(cache_file)
    return None


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from model response, handling common formatting issues
    
    Args:
        response: Raw model response
        
    Returns:
        Parsed JSON dictionary
    """
    # Remove markdown code blocks if present
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    # Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON from response: {response}") from e


def extract_choice_letter(response: str) -> str:
    """
    Extract the choice letter (A, B, C, or D) from model response
    
    Args:
        response: Raw model response
        
    Returns:
        Choice letter (A, B, C, or D)
    """
    response = response.strip().upper()
    
    # Check if response is just the letter
    if response in ['A', 'B', 'C', 'D']:
        return response
    
    # Try to find the letter in the response
    import re
    # Look for patterns like "A)", "A.", "A:", "(A)", "[A]", "Answer: A", etc.
    patterns = [
        r'\b([ABCD])\)',
        r'\b([ABCD])\.',
        r'\b([ABCD]):',
        r'\(([ABCD])\)',
        r'\[([ABCD])\]',
        r'answer[:\s]+([ABCD])',
        r'choice[:\s]+([ABCD])',
        r'\b([ABCD])\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no letter found, return the first character if it's A-D
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    
    raise ValueError(f"Could not extract choice letter from response: {response}")


def format_choices(choices: list) -> str:
    """
    Format multiple choice options
    
    Args:
        choices: List of choice strings
        
    Returns:
        Formatted string with lettered options
    """
    letters = ['A', 'B', 'C', 'D']
    return '\n'.join([f"{letters[i]}) {choice}" for i, choice in enumerate(choices)])


def choices_to_letter(choices: list, index: int) -> str:
    """
    Convert choice index to letter
    
    Args:
        choices: List of choices
        index: Index of the correct choice
        
    Returns:
        Letter (A, B, C, or D)
    """
    letters = ['A', 'B', 'C', 'D']
    return letters[index]


def letter_to_index(letter: str) -> int:
    """
    Convert choice letter to index
    
    Args:
        letter: Choice letter (A, B, C, or D)
        
    Returns:
        Index (0, 1, 2, or 3)
    """
    letters = ['A', 'B', 'C', 'D']
    return letters.index(letter.upper())
