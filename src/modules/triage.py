"""
Module 1: Triage Router
Classifies questions by complexity and knowledge requirements
"""
import re
from pathlib import Path
from typing import Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.ollama_wrapper import OllamaTextModel
from prompts.prompt_templates import format_triage_prompt
from utils import setup_logging, parse_json_response
from config import TEMPERATURE_CONFIG

logger = setup_logging(__name__)


class TriageRouter:
    """
    Triage Router for classifying VQA questions
    
    Classifies questions along two dimensions:
    1. Complexity: ATOMIC (single-step) vs COMPOSITIONAL (multi-step)
    2. Knowledge: VISUAL (from image) vs KNOWLEDGE-BASED (external facts)
    """
    
    def __init__(self):
        """Initialize the triage router with text model"""
        self.text_model = OllamaTextModel()
        logger.info("Initialized Triage Router")
    
    @staticmethod
    def _strip_think_tags(response: str) -> str:
        """
        Remove Qwen's <think>...</think> reasoning tokens from response
        
        Args:
            response: Raw model response
            
        Returns:
            Response with think tags removed
        """
        # Remove everything between <think> and </think> tags
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()
    
    def classify(self, question: str, use_cache: bool = True) -> Dict[str, str]:
        """
        Classify a question's properties
        
        Args:
            question: The question text to classify
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary with 'complexity' and 'knowledge' keys
            Example: {"complexity": "ATOMIC", "knowledge": "VISUAL"}
        """
        logger.debug(f"Classifying question: {question}")
        
        # Get the prompt
        prompt = format_triage_prompt(question)
        
        # Call the text model with temperature=0 for deterministic output
        response = self.text_model.generate(
            prompt=prompt,
            temperature=TEMPERATURE_CONFIG['triage'],
            use_cache=use_cache
        )
        
        logger.debug(f"Raw triage response: {response}")
        
        # Strip think tags from Qwen model
        cleaned_response = self._strip_think_tags(response)
        logger.debug(f"Cleaned response (think tags removed): {cleaned_response}")
        
        # Parse JSON response
        try:
            result = parse_json_response(cleaned_response)
            
            # Validate the response
            if 'complexity' not in result or 'knowledge' not in result:
                raise ValueError(f"Missing required keys in triage response: {result}")
            
            # Validate values
            valid_complexity = ['ATOMIC', 'COMPOSITIONAL']
            valid_knowledge = ['VISUAL', 'KNOWLEDGE-BASED']
            
            if result['complexity'] not in valid_complexity:
                logger.warning(
                    f"Invalid complexity value: {result['complexity']}. "
                    f"Expected one of {valid_complexity}. Defaulting to ATOMIC."
                )
                result['complexity'] = 'ATOMIC'
            
            if result['knowledge'] not in valid_knowledge:
                logger.warning(
                    f"Invalid knowledge value: {result['knowledge']}. "
                    f"Expected one of {valid_knowledge}. Defaulting to VISUAL."
                )
                result['knowledge'] = 'VISUAL'
            
            logger.info(
                f"Question classified as: "
                f"Complexity={result['complexity']}, "
                f"Knowledge={result['knowledge']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse triage response: {e}")
            logger.error(f"Response was: {cleaned_response}")
            # Return default classification
            default = {"complexity": "ATOMIC", "knowledge": "VISUAL"}
            logger.warning(f"Using default classification: {default}")
            return default


def test_triage():
    """Test the triage router with sample questions"""
    logger.info("="*60)
    logger.info("Testing Triage Router")
    logger.info("="*60)
    
    triage = TriageRouter()
    
    # Test cases with expected classifications
    test_cases = [
        {
            "question": "What color is the car?",
            "expected": {"complexity": "ATOMIC", "knowledge": "VISUAL"}
        },
        {
            "question": "In which city is this famous landmark located?",
            "expected": {"complexity": "ATOMIC", "knowledge": "KNOWLEDGE-BASED"}
        },
        {
            "question": "What is the brand of the laptop next to the red book?",
            "expected": {"complexity": "COMPOSITIONAL", "knowledge": "VISUAL"}
        },
        {
            "question": "Who is the manufacturer of the vehicle parked in front of the building?",
            "expected": {"complexity": "COMPOSITIONAL", "knowledge": "KNOWLEDGE-BASED"}
        },
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Case {i}/{len(test_cases)}")
        logger.info(f"{'='*60}")
        logger.info(f"Question: {test_case['question']}")
        logger.info(f"Expected: {test_case['expected']}")
        
        result = triage.classify(test_case['question'])
        logger.info(f"Actual:   {result}")
        
        # Check if correct
        matches = (
            result['complexity'] == test_case['expected']['complexity'] and
            result['knowledge'] == test_case['expected']['knowledge']
        )
        
        status = "✓ MATCH" if matches else "✗ MISMATCH"
        logger.info(f"Status:   {status}")
        
        results.append({
            "question": test_case['question'],
            "expected": test_case['expected'],
            "actual": result,
            "matches": matches
        })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    matches = sum(1 for r in results if r['matches'])
    logger.info(f"Matched: {matches}/{len(results)}")
    logger.info(f"Accuracy: {matches/len(results)*100:.1f}%")
    
    if matches == len(results):
        logger.info("✓ All tests passed!")
    else:
        logger.warning("⚠ Some tests failed")
        for r in results:
            if not r['matches']:
                logger.warning(f"  Failed: {r['question']}")
    
    logger.info(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    test_triage()
