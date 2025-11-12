"""
Baseline VQA Implementation
Naive end-to-end prompting with vision model
"""
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.ollama_wrapper import OllamaVisionModel
from prompts.prompt_templates import format_baseline_prompt
from data.dataset_loader import AOKVQASample
from utils import (
    setup_logging,
    format_choices,
    extract_choice_letter,
    letter_to_index,
)
from config import TEMPERATURE_CONFIG

logger = setup_logging(__name__)


class BaselineVQA:
    """Baseline VQA system using naive end-to-end prompting"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize baseline VQA system
        
        Args:
            model_name: Ollama vision model name (default from config)
        """
        self.vision_model = OllamaVisionModel(model_name) if model_name else OllamaVisionModel()
        logger.info(f"Initialized Baseline VQA with model: {self.vision_model.model_name}")
    
    def answer_question(
        self,
        sample: AOKVQASample,
        image_path: Path,
        temperature: float = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a question using naive end-to-end prompting
        
        Args:
            sample: AOKVQASample object
            image_path: Path to the image
            temperature: Sampling temperature (default from config)
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary containing prediction and metadata
        """
        if temperature is None:
            temperature = TEMPERATURE_CONFIG['executor']
        
        # Format the prompt with choices
        choices_text = format_choices(sample.choices)
        prompt = format_baseline_prompt(sample.question, choices_text)
        
        logger.debug(f"Answering question: {sample.question}")
        logger.debug(f"Prompt: {prompt}")
        
        # Get model response
        try:
            # print("[DEBUG] Calling vision model generate...")
            response = self.vision_model.generate(
                prompt=prompt,
                image_path=image_path,
                temperature=temperature,
                use_cache=use_cache,
            )
            
            # print(f"[DEBUG] Raw response: {response}")
            
            # Extract the choice letter
            predicted_letter = extract_choice_letter(response)
            predicted_idx = letter_to_index(predicted_letter)
            
            # Check if correct
            is_correct = (predicted_idx == sample.correct_choice_idx)
            
            result = {
                'question_id': sample.question_id,
                'question': sample.question,
                'choices': sample.choices,
                'predicted_letter': predicted_letter,
                'predicted_idx': predicted_idx,
                'predicted_choice': sample.choices[predicted_idx],
                'correct_letter': sample.correct_choice_letter,
                'correct_idx': sample.correct_choice_idx,
                'correct_choice': sample.correct_choice,
                'is_correct': is_correct,
                'raw_response': response,
            }
            
            logger.info(
                f"Question {sample.question_id}: "
                f"Predicted {predicted_letter} ({sample.choices[predicted_idx]}), "
                f"Correct {sample.correct_choice_letter} ({sample.correct_choice}) - "
                f"{'✓' if is_correct else '✗'}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering question {sample.question_id}: {e}")
            return {
                'question_id': sample.question_id,
                'question': sample.question,
                'error': str(e),
                'is_correct': False,
            }
    
    def evaluate(
        self,
        samples: List[AOKVQASample],
        image_getter,
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the baseline on a list of samples
        
        Args:
            samples: List of AOKVQASample objects
            image_getter: Function that takes a sample and returns image path
            temperature: Sampling temperature
            
        Returns:
            Dictionary with results and metrics
        """
        results = []
        correct_count = 0
        
        logger.info(f"Starting baseline evaluation on {len(samples)} samples")
        
        for i, sample in enumerate(samples):
            logger.info(f"\n{'='*20}")
            logger.info(f"Sample {i+1}/{len(samples)}")
            logger.info(f"{'='*20}")
            
            image_path = image_getter(sample)
            result = self.answer_question(sample, image_path, temperature)
            
            if result.get('is_correct', False):
                correct_count += 1
            
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                current_acc = correct_count / (i + 1) * 100
                logger.info(f"\nProgress: {i+1}/{len(samples)} - Accuracy: {current_acc:.2f}%")
        
        # Calculate final metrics
        accuracy = correct_count / len(samples) * 100
        
        evaluation_result = {
            'method': 'baseline',
            'model': self.vision_model.model_name,
            'total_samples': len(samples),
            'correct': correct_count,
            'incorrect': len(samples) - correct_count,
            'accuracy': accuracy,
            'results': results,
        }
        
        logger.info(f"\n{'='*20}")
        logger.info(f"BASELINE EVALUATION COMPLETE")
        logger.info(f"{'='*20}")
        logger.info(f"Total Samples: {len(samples)}")
        logger.info(f"Correct: {correct_count}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"{'='*20}\n")
        
        return evaluation_result


def main():
    """Test the baseline system"""
    from data.dataset_loader import AOKVQADataset
    from config import RESULTS_DIR
    
    logger.info("Testing Baseline VQA System")
    
    # Load dataset
    dataset = AOKVQADataset()
    
    # Load a small subset for testing
    samples = dataset.load_split('val')[:5]  # Just test on 5 samples
    
    logger.info(f"Testing on {len(samples)} samples")
    
    # Initialize baseline
    baseline = BaselineVQA()
    
    # Define image getter function
    def get_image_path(sample):
        return dataset.get_image_path(sample.split, sample.image_id)
    
    # Evaluate
    results = baseline.evaluate(samples, get_image_path)
    
    # Save results
    import json
    output_file = RESULTS_DIR / "baseline_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
