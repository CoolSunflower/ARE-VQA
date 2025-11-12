"""
Module 3: Query Planner
Decomposes complex questions into simpler sub-questions
"""
import re
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.ollama_wrapper import OllamaTextModel
from prompts.prompt_templates import format_planner_prompt
from utils import setup_logging
from config import TEMPERATURE_CONFIG

logger = setup_logging(__name__)


class QueryPlanner:
    """
    Query Planner for decomposing complex VQA questions
    
    Breaks down COMPOSITIONAL questions into simpler, atomic sub-questions
    that can be answered sequentially.
    """
    
    def __init__(self):
        """Initialize the query planner with text model"""
        self.text_model = OllamaTextModel()
        logger.info("Initialized Query Planner")
    
    @staticmethod
    def _strip_think_tags(response: str) -> str:
        """Remove Qwen's <think>...</think> reasoning tokens"""
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()
    
    def plan(
        self,
        question: str,
        context: str,
        triage_output: Dict[str, str],
        config: Dict[str, bool],
        use_cache: bool = True
    ) -> List[str]:
        """
        Plan the question decomposition
        
        Args:
            question: The original question
            context: Context (visual + knowledge) from Context Builder
            triage_output: Triage classification with 'complexity' key
            config: Configuration dict with 'use_planner' flag
            use_cache: Whether to use cached responses
            
        Returns:
            List of sub-questions. For ATOMIC questions or when planner is disabled,
            returns [original_question]. For COMPOSITIONAL, returns list of sub-questions.
        """
        # Check if we should decompose
        should_decompose = (
            config.get("use_planner", False) and
            triage_output.get("complexity") == "COMPOSITIONAL"
        )
        
        if not should_decompose:
            logger.info(f"Skipping decomposition (ATOMIC or planner disabled)")
            return [question]
        
        logger.info(f"Decomposing COMPOSITIONAL question: {question}")
        
        # Get decomposition from LLM
        prompt = format_planner_prompt(question, context)
        
        response = self.text_model.generate(
            prompt=prompt,
            temperature=TEMPERATURE_CONFIG['planner'],
            use_cache=use_cache
        )
        
        logger.debug(f"Raw planner response: {response}")
        
        # Strip think tags
        cleaned_response = self._strip_think_tags(response)
        logger.debug(f"Cleaned response: {cleaned_response}")
        
        # Parse numbered list
        sub_questions = self._parse_sub_questions(cleaned_response, question)
        
        logger.info(f"Decomposed into {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            logger.info(f"  {i}. {sq}")
        
        return sub_questions
    
    def _parse_sub_questions(self, response: str, original_question: str) -> List[str]:
        """
        Parse numbered list of sub-questions from LLM response
        
        Args:
            response: Cleaned LLM response
            original_question: Fallback if parsing fails
            
        Returns:
            List of sub-question strings
        """
        # Try to find numbered list patterns
        # Patterns: "1. Question", "1) Question", "1: Question", etc.
        patterns = [
            r'^\d+[\.\)]\s*(.+)$',  # 1. or 1)
            r'^\d+:\s*(.+)$',        # 1:
            r'^-\s*(.+)$',           # - bullet
        ]
        
        lines = response.split('\n')
        sub_questions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    sub_q = match.group(1).strip()
                    if sub_q and len(sub_q) > 3:  # Filter out very short matches
                        sub_questions.append(sub_q)
                    break
        
        # Validation: if we got no sub-questions or parsing failed, return original
        if not sub_questions:
            logger.warning("Failed to parse sub-questions from response, using original question")
            return [original_question]
        
        # Validation: filter out duplicates
        seen = set()
        filtered = []
        for sq in sub_questions:
            sq_lower = sq.lower()
            if sq_lower not in seen:
                seen.add(sq_lower)
                filtered.append(sq)
        
        return filtered if filtered else [original_question]


def test_planner():
    """Test the query planner with sample questions"""
    logger.info("="*60)
    logger.info("Testing Query Planner")
    logger.info("="*60)
    
    planner = QueryPlanner()
    
    # Test cases
    test_cases = [
        {
            "question": "What color is the car?",
            "triage": {"complexity": "ATOMIC", "knowledge": "VISUAL"},
            "context": "VISUAL CONTEXT:\nA red car is parked on the street.",
            "config": {"use_planner": True},
            "expected_count": 1  # Should NOT decompose (ATOMIC)
        },
        {
            "question": "What is the brand of the laptop next to the red book?",
            "triage": {"complexity": "COMPOSITIONAL", "knowledge": "VISUAL"},
            "context": "VISUAL CONTEXT:\nA desk with a laptop and a red book.",
            "config": {"use_planner": True},
            "expected_count": "2-3"  # Should decompose (COMPOSITIONAL)
        },
        {
            "question": "Who manufactured the vehicle in front of the Eiffel Tower?",
            "triage": {"complexity": "COMPOSITIONAL", "knowledge": "KNOWLEDGE-BASED"},
            "context": "VISUAL CONTEXT:\nA car in front of a large tower.\n\nEXTERNAL KNOWLEDGE:\nThe Eiffel Tower is in Paris, France.",
            "config": {"use_planner": True},
            "expected_count": "2-3"  # Should decompose
        },
        {
            "question": "What is the brand of the laptop next to the red book?",
            "triage": {"complexity": "COMPOSITIONAL", "knowledge": "VISUAL"},
            "context": "VISUAL CONTEXT:\nA desk with a laptop and a red book.",
            "config": {"use_planner": False},
            "expected_count": 1  # Should NOT decompose (planner disabled)
        },
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Case {i}/{len(test_cases)}")
        logger.info(f"{'='*60}")
        logger.info(f"Question: {test_case['question']}")
        logger.info(f"Triage: {test_case['triage']}")
        logger.info(f"Config: {test_case['config']}")
        logger.info(f"Expected sub-questions: {test_case['expected_count']}")
        
        sub_questions = planner.plan(
            test_case['question'],
            test_case['context'],
            test_case['triage'],
            test_case['config'],
            use_cache=False  # Fresh calls for testing
        )
        
        logger.info(f"\nActual: {len(sub_questions)} sub-questions")
        for j, sq in enumerate(sub_questions, 1):
            logger.info(f"  {j}. {sq}")
        
        # Check if count matches expectation
        expected = test_case['expected_count']
        if isinstance(expected, str) and '-' in expected:
            # Range check (e.g., "2-3")
            min_exp, max_exp = map(int, expected.split('-'))
            matches = min_exp <= len(sub_questions) <= max_exp
        else:
            matches = len(sub_questions) == expected
        
        status = "✓ PASS" if matches else "✗ FAIL"
        logger.info(f"Status: {status}")
        
        results.append({
            "question": test_case['question'],
            "expected": expected,
            "actual": len(sub_questions),
            "matches": matches
        })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    passed = sum(1 for r in results if r['matches'])
    logger.info(f"Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        logger.info("✓ All tests passed!")
    else:
        logger.warning("⚠ Some tests failed")
    
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    test_planner()
