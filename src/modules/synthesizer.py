"""
Module 5: Synthesizer
Combines intermediate answers and selects final multiple choice answer
"""
from pathlib import Path
from typing import List, Dict
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))

from models.ollama_wrapper import OllamaTextModel
from prompts.prompt_templates import format_synthesizer_prompt
from utils import setup_logging, extract_choice_letter, format_choices
from config import TEMPERATURE_CONFIG

logger = setup_logging(__name__)


class Synthesizer:
    """
    Synthesizer for combining intermediate answers into final MC selection
    
    Takes intermediate answers from executor and synthesizes final answer.
    """
    
    def __init__(self):
        """Initialize the synthesizer with text model"""
        self.text_model = OllamaTextModel()
        logger.info("Initialized Synthesizer")
    
    def _strip_think_tags(self, response: str) -> str:
        """
        Strip <think>...</think> tags from Qwen's response
        
        Args:
            response: Raw response from model
            
        Returns:
            Cleaned response without think tags
        """
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()
    
    def synthesize(
        self,
        question: str,
        context: str,
        intermediate_answers: List[str],
        choices: List[str],
        use_cache: bool = True
    ) -> str:
        """
        Synthesize final answer from intermediate answers
        
        Args:
            question: Original question
            context: Context from Context Builder
            intermediate_answers: List of intermediate answers from executor
            choices: List of MC choices (4 options)
            use_cache: Whether to use cached responses
            
        Returns:
            Final choice letter (A, B, C, or D)
        """
        logger.info(f"Synthesizing final answer")
        logger.info(f"Question: {question}")
        logger.info(f"Intermediate answers: {len(intermediate_answers)}")
        
        # Format intermediate answers as numbered list
        if len(intermediate_answers) == 1:
            # Single answer (atomic question)
            intermediate_text = intermediate_answers[0]
        else:
            # Multiple answers (compositional question)
            intermediate_text = "\n".join([
                f"{i}. {ans}"
                for i, ans in enumerate(intermediate_answers, 1)
            ])
        
        logger.info(f"\nIntermediate Answers:\n{intermediate_text}\n")
        
        # Format choices
        choices_text = format_choices(choices)
        
        # Format synthesizer prompt
        prompt = format_synthesizer_prompt(
            question,
            context,
            intermediate_text,
            choices_text
        )
        
        # Get response from LLM
        response = self.text_model.generate(
            prompt,
            temperature=TEMPERATURE_CONFIG['synthesizer'],
            use_cache=use_cache
        )
        
        # Strip <think> tags
        cleaned_response = self._strip_think_tags(response)
        
        logger.info(f"\nRaw Response: {response[:200]}...")
        logger.info(f"Cleaned Response: {cleaned_response[:200]}...")
        
        # Extract choice letter
        choice = extract_choice_letter(cleaned_response)
        
        if choice:
            logger.info(f"\n✓ Synthesized final answer: {choice}")
        else:
            logger.warning(f"\n⚠ Failed to extract choice, defaulting to 'A'")
            choice = 'A'
        
        return choice


def test_synthesizer():
    """Test the synthesizer with sample intermediate answers"""
    logger.info("="*60)
    logger.info("Testing Synthesizer")
    logger.info("="*60)
    
    synthesizer = Synthesizer()
    
    # Test 1: ATOMIC question (single intermediate answer)
    logger.info(f"\n{'='*60}")
    logger.info("Test 1: ATOMIC Question (Single Intermediate Answer)")
    logger.info(f"{'='*60}")
    
    question1 = "What is in the motorcyclist's mouth?"
    context1 = "VISUAL CONTEXT:\nA motorcyclist riding a bike on the street wearing protective gear."
    intermediate1 = ["The motorcyclist appears to have a cigarette in their mouth."]
    choices1 = ["cigarette", "thirty", "dirty", "icing"]
    
    logger.info(f"\nQuestion: {question1}")
    logger.info(f"Choices: {choices1}")
    logger.info(f"Intermediate: {intermediate1[0]}")
    
    answer1 = synthesizer.synthesize(
        question1,
        context1,
        intermediate1,
        choices1,
        use_cache=False
    )
    
    logger.info(f"\n→ Final Answer: {answer1}")
    logger.info(f"   Expected: A (cigarette)")
    
    # Test 2: COMPOSITIONAL question (multiple intermediate answers)
    logger.info(f"\n{'='*60}")
    logger.info("Test 2: COMPOSITIONAL Question (Multiple Intermediate Answers)")
    logger.info(f"{'='*60}")
    
    question2 = "What is the brand of the laptop next to the red book?"
    context2 = """VISUAL CONTEXT:
A desk setup with multiple items including books and electronic devices.

KNOWLEDGE:
Common laptop brands include Apple, Dell, HP, and Lenovo. Apple laptops have distinctive 
designs with their logo clearly visible."""
    intermediate2 = [
        "Yes, there is a red book visible on the desk.",
        "Next to the red book there is a laptop computer.",
        "The laptop appears to be an Apple MacBook based on its design and visible logo."
    ]
    choices2 = ["Apple", "Microsoft", "Sony", "Samsung"]
    
    logger.info(f"\nQuestion: {question2}")
    logger.info(f"Choices: {choices2}")
    logger.info(f"Intermediate answers: {len(intermediate2)}")
    for i, ans in enumerate(intermediate2, 1):
        logger.info(f"  {i}. {ans}")
    
    answer2 = synthesizer.synthesize(
        question2,
        context2,
        intermediate2,
        choices2,
        use_cache=False
    )
    
    logger.info(f"\n→ Final Answer: {answer2}")
    logger.info(f"   Expected: A (Apple)")
    
    # Test 3: KNOWLEDGE-BASED question
    logger.info(f"\n{'='*60}")
    logger.info("Test 3: KNOWLEDGE-BASED Question")
    logger.info(f"{'='*60}")
    
    question3 = "In which country is this famous landmark located?"
    context3 = """VISUAL CONTEXT:
A large iconic tower structure with distinctive lattice metalwork design, photographed 
against a blue sky.

KNOWLEDGE:
The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was 
constructed in 1889 and has become one of the most recognizable structures in the world."""
    intermediate3 = [
        "Based on the visual appearance and the knowledge that this is the Eiffel Tower, it is located in France."
    ]
    choices3 = ["Germany", "France", "Italy", "Spain"]
    
    logger.info(f"\nQuestion: {question3}")
    logger.info(f"Choices: {choices3}")
    logger.info(f"Intermediate: {intermediate3[0]}")
    
    answer3 = synthesizer.synthesize(
        question3,
        context3,
        intermediate3,
        choices3,
        use_cache=False
    )
    
    logger.info(f"\n→ Final Answer: {answer3}")
    logger.info(f"   Expected: B (France)")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Synthesizer Test Complete")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    test_synthesizer()
