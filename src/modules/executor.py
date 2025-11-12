"""
Module 4: Tool Executor
Answers atomic sub-questions using the vision model
"""
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.ollama_wrapper import OllamaVisionModel
from prompts.prompt_templates import format_executor_prompt
from utils import setup_logging, format_choices
from config import TEMPERATURE_CONFIG

logger = setup_logging(__name__)


class ToolExecutor:
    """
    Tool Executor for answering sub-questions
    
    Executes the vision model on each sub-question to get intermediate answers.
    """
    
    def __init__(self):
        """Initialize the tool executor with vision model"""
        self.vision_model = OllamaVisionModel()
        logger.info("Initialized Tool Executor")
    
    def execute(
        self,
        image_path: Path,
        sub_questions: List[str],
        context: str,
        choices: List[str],
        config: Dict[str, bool],
        use_cache: bool = True
    ) -> List[str]:
        """
        Execute answering for all sub-questions
        
        Args:
            image_path: Path to the image
            sub_questions: List of sub-questions from planner
            context: Context (visual + knowledge) from Context Builder
            choices: List of MC choices (only used for atomic questions)
            config: Configuration dict (not currently used by executor)
            use_cache: Whether to use cached responses
            
        Returns:
            List of answer strings, one per sub-question
        """
        logger.info(f"Executing {len(sub_questions)} sub-questions")
        
        # Only use MC format if this is a single atomic question
        # For decomposed sub-questions, we want open-ended answers to build context
        use_mc_format = (len(sub_questions) == 1)
        
        answers = []
        for i, sub_question in enumerate(sub_questions, 1):
            logger.info(f"\nAnswering sub-question {i}/{len(sub_questions)}: {sub_question}")
            
            answer = self._answer_sub_question(
                image_path,
                sub_question,
                context,
                choices if use_mc_format else None,  # Only pass choices for atomic questions
                use_cache
            )
            
            answers.append(answer)
            logger.info(f"Answer {i}: {answer[:100]}...")
        
        logger.info(f"\nExecuted all {len(sub_questions)} sub-questions")
        return answers
    
    def _answer_sub_question(
        self,
        image_path: Path,
        sub_question: str,
        context: str,
        choices: List[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Answer a single sub-question using the vision model
        
        Args:
            image_path: Path to the image
            sub_question: The sub-question to answer
            context: Context from Context Builder (KNOWLEDGE ONLY - visual context removed)
            choices: List of MC choices (only for atomic questions, None for sub-questions)
            use_cache: Whether to use cached responses
            
        Returns:
            Answer string
        """
        # Extract only KNOWLEDGE section from context (skip VISUAL CONTEXT to avoid bias)
        # Vision model should see the image directly, not text description
        knowledge_context = ""
        if "KNOWLEDGE:" in context:
            # Extract just the knowledge part
            parts = context.split("KNOWLEDGE:")
            if len(parts) > 1:
                knowledge_context = "KNOWLEDGE:\n" + parts[1].strip()
        
        # Format prompt - with or without MC choices
        if choices:
            # Atomic question: use MC format
            choices_text = format_choices(choices)
            prompt = format_executor_prompt(sub_question, knowledge_context, choices_text)
            max_tokens = None  # No limit needed for MC (just returns a letter)
        else:
            # Sub-question from decomposition: open-ended format
            prompt = format_executor_prompt(sub_question, knowledge_context, None)
            max_tokens = 200  # Limit to ~100 tokens for concise descriptive answers
        
        # Get answer from vision model
        answer = self.vision_model.generate(
            prompt=prompt,
            image_path=image_path,
            temperature=TEMPERATURE_CONFIG['executor'],
            use_cache=use_cache,
            max_tokens=max_tokens
        )
        
        return answer.strip()


def test_executor():
    """Test the tool executor with sample sub-questions"""
    from data.dataset_loader import AOKVQADataset
    
    logger.info("="*60)
    logger.info("Testing Tool Executor")
    logger.info("="*60)
    
    executor = ToolExecutor()
    dataset = AOKVQADataset()
    
    # Load a sample
    sample = dataset.load_split('val')[0]
    image_path = dataset.get_image_path(sample.split, sample.image_id)
    
    logger.info(f"\nTest Question: {sample.question}")
    logger.info(f"Correct Answer: {sample.correct_choice}")
    
    # Test 1: Single atomic question
    logger.info(f"\n{'='*60}")
    logger.info("Test 1: ATOMIC Question (Single Sub-Question)")
    logger.info(f"{'='*60}")
    
    context1 = "VISUAL CONTEXT:\nA motorcyclist riding a bike on the street."
    sub_questions1 = [sample.question]
    choices1 = sample.choices
    
    answers1 = executor.execute(
        image_path,
        sub_questions1,
        context1,
        choices1,
        config={},
        use_cache=False
    )
    
    logger.info(f"\nSub-questions: {len(sub_questions1)}")
    logger.info(f"Answers: {len(answers1)}")
    for i, (sq, ans) in enumerate(zip(sub_questions1, answers1), 1):
        logger.info(f"\n{i}. Q: {sq}")
        logger.info(f"   A: {ans}")
    
    # Test 2: Multiple sub-questions (simulated compositional)
    logger.info(f"\n{'='*60}")
    logger.info("Test 2: COMPOSITIONAL Question (Multiple Sub-Questions)")
    logger.info(f"{'='*60}")
    
    context2 = "VISUAL CONTEXT:\nA motorcyclist with a helmet riding on the street."
    sub_questions2 = [
        "Is there a motorcyclist in the image?",
        "What is in the motorcyclist's mouth?",
    ]
    choices2 = sample.choices  # Same choices for guidance
    
    answers2 = executor.execute(
        image_path,
        sub_questions2,
        context2,
        choices2,
        config={},
        use_cache=False
    )
    
    logger.info(f"\nSub-questions: {len(sub_questions2)}")
    logger.info(f"Answers: {len(answers2)}")
    for i, (sq, ans) in enumerate(zip(sub_questions2, answers2), 1):
        logger.info(f"\n{i}. Q: {sq}")
        logger.info(f"   A: {ans}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Tool Executor Test Complete")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    test_executor()
