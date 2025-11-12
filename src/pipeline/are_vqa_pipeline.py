"""
ARE-VQA Pipeline Orchestrator
Assembles the complete 5-module pipeline with ablation support
"""
from pathlib import Path
from typing import Dict, List
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from modules.triage import TriageRouter
from modules.context_builder import ContextBuilder
from modules.planner import QueryPlanner
from modules.executor import ToolExecutor
from modules.synthesizer import Synthesizer
from data.dataset_loader import AOKVQADataset, AOKVQASample
from utils import setup_logging, format_choices

logger = setup_logging(__name__)


class AREVQAPipeline:
    """
    Complete ARE-VQA Pipeline
    
    Pipeline Stages:
    1. Triage Router - Classify question complexity and knowledge needs
    2. Context Builder - Extract visual context + retrieve external knowledge
    3. Query Planner - Decompose compositional questions into sub-questions
    4. Tool Executor - Answer sub-questions using vision model
    5. Synthesizer - Combine intermediate answers into final MC selection
    
    Configuration options:
    - use_planner: Enable/disable query decomposition (default: True)
    - use_knowledge: Enable/disable external knowledge retrieval (default: True)
    """
    
    def __init__(self, config: Dict[str, bool] = None):
        """
        Initialize the ARE-VQA pipeline
        
        Args:
            config: Configuration dict with options:
                - use_planner (bool): Enable query decomposition (default: True)
                - use_knowledge (bool): Enable knowledge retrieval (default: True)
        """
        self.config = config or {"use_planner": True, "use_knowledge": True}
        
        logger.info("="*60)
        logger.info("Initializing ARE-VQA Pipeline")
        logger.info(f"Configuration: {self.config}")
        logger.info("="*60)
        
        # Initialize all modules
        self.triage = TriageRouter()
        self.context_builder = ContextBuilder()
        self.planner = QueryPlanner()
        self.executor = ToolExecutor()
        self.synthesizer = Synthesizer()
        
        logger.info("✓ All modules initialized")
    
    def answer_question(
        self,
        sample: AOKVQASample,
        image_path: Path,
        use_cache: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Answer a single VQA question through the complete pipeline
        
        Args:
            sample: AOKVQASample with question, choices, etc.
            image_path: Path to the image
            use_cache: Whether to use cached model responses
            verbose: Whether to log detailed trace information
            
        Returns:
            Dict with:
                - predicted_choice: Final predicted choice letter (A/B/C/D)
                - correct_choice: Ground truth choice letter
                - is_correct: Whether prediction matches ground truth
                - trace: Dict with intermediate outputs from each module
        """
        if verbose:
            logger.info("\n" + "="*60)
            logger.info(f"Processing Question: {sample.question}")
            logger.info(f"Choices: {sample.choices}")
            logger.info("="*60)
        
        trace = {}
        
        # Stage 1: Triage Router
        if verbose:
            logger.info("\n[Stage 1] Triage Router")
        
        triage_output = self.triage.classify(
            sample.question,
            use_cache=use_cache
        )
        trace['triage'] = triage_output
        
        if verbose:
            logger.info(f"  Complexity: {triage_output['complexity']}")
            logger.info(f"  Knowledge: {triage_output['knowledge']}")
        
        # Stage 2: Context Builder
        if verbose:
            logger.info("\n[Stage 2] Context Builder")
        
        context = self.context_builder.build(
            question=sample.question,
            image_path=image_path,
            triage_output=triage_output,
            config=self.config,
            use_cache=use_cache
        )
        trace['context'] = context
        
        if verbose:
            logger.info(f"  Context length: {len(context)} chars")
            logger.info(f"  Preview: {context[:200]}...")
        
        # Stage 3: Query Planner
        if verbose:
            logger.info("\n[Stage 3] Query Planner")
        
        sub_questions = self.planner.plan(
            question=sample.question,
            context=context,
            triage_output=triage_output,
            config=self.config,
            use_cache=use_cache
        )
        trace['sub_questions'] = sub_questions
        
        if verbose:
            logger.info(f"  Sub-questions: {len(sub_questions)}")
            for i, sq in enumerate(sub_questions, 1):
                logger.info(f"    {i}. {sq}")
        
        # Stage 4: Tool Executor
        if verbose:
            logger.info("\n[Stage 4] Tool Executor")
        
        intermediate_answers = self.executor.execute(
            image_path=image_path,
            sub_questions=sub_questions,
            context=context,
            choices=sample.choices,
            config=self.config,
            use_cache=use_cache
        )
        trace['intermediate_answers'] = intermediate_answers
        
        if verbose:
            logger.info(f"  Intermediate answers: {len(intermediate_answers)}")
            for i, ans in enumerate(intermediate_answers, 1):
                logger.info(f"    {i}. {ans[:100]}...")
        
        # Stage 5: Synthesizer
        if verbose:
            logger.info("\n[Stage 5] Synthesizer")
        
        # Optimization: If we have a single atomic question and executor returned a letter,
        # skip synthesizer (it's redundant and can introduce errors)
        if len(sub_questions) == 1 and len(intermediate_answers[0].strip()) <= 3:
            # Single letter answer (likely "A", "B", "C", or "D")
            predicted_choice = intermediate_answers[0].strip()
            if verbose:
                logger.info(f"  Skipping synthesizer (single atomic answer: {predicted_choice})")
        else:
            # Use synthesizer for compositional questions or complex answers
            predicted_choice = self.synthesizer.synthesize(
                question=sample.question,
                context=context,
                intermediate_answers=intermediate_answers,
                choices=sample.choices,
                use_cache=use_cache
            )
        
        if verbose:
            logger.info(f"  Predicted: {predicted_choice}")
            logger.info(f"  Correct: {sample.correct_choice_letter}")
        
        # Prepare result
        is_correct = predicted_choice == sample.correct_choice_letter
        
        if verbose:
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            logger.info(f"\n{status}")
            logger.info("="*60)
        
        return {
            'question_id': sample.question_id,
            'predicted_choice': predicted_choice,
            'correct_choice': sample.correct_choice,
            'correct_choice_letter': sample.correct_choice_letter,
            'is_correct': is_correct,
            'trace': trace
        }
    
    def evaluate(
        self,
        samples: List[AOKVQASample],
        dataset: AOKVQADataset,
        use_cache: bool = True,
        verbose: bool = False,
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate the pipeline on multiple samples
        
        Args:
            samples: List of AOKVQASample objects
            dataset: AOKVQADataset instance for getting image paths
            use_cache: Whether to use cached model responses
            verbose: Whether to log detailed trace for each question
            save_results: Whether to save results to JSON file
            
        Returns:
            Dict with:
                - accuracy: Overall accuracy
                - total: Total questions
                - correct: Number correct
                - results: List of per-question results
        """
        logger.info("\n" + "="*60)
        logger.info(f"Evaluating ARE-VQA Pipeline on {len(samples)} samples")
        logger.info(f"Configuration: {self.config}")
        logger.info("="*60)
        
        results = []
        correct = 0
        
        for i, sample in enumerate(samples, 1):
            logger.info(f"\n[{i}/{len(samples)}] Question ID: {sample.question_id}")
            
            image_path = dataset.get_image_path(sample.split, sample.image_id)
            
            result = self.answer_question(
                sample,
                image_path,
                use_cache=use_cache,
                verbose=verbose
            )
            
            results.append(result)
            
            if result['is_correct']:
                correct += 1
            
            # Log progress
            if not verbose:
                status = "✓" if result['is_correct'] else "✗"
                logger.info(f"  {status} Predicted: {result['predicted_choice']}, Correct: {result['correct_choice']}")
            
            accuracy = correct / i
            logger.info(f"  Running accuracy: {accuracy:.2%} ({correct}/{i})")
        
        # Final summary
        final_accuracy = correct / len(samples)
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total questions: {len(samples)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {final_accuracy:.2%}")
        logger.info("="*60)
        
        summary = {
            'config': self.config,
            'total': len(samples),
            'correct': correct,
            'accuracy': final_accuracy,
            'results': results
        }
        
        # Save results
        if save_results:
            self._save_results(summary)
        
        return summary
    
    def _save_results(self, summary: Dict):
        """Save evaluation results to JSON file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Create filename with config and timestamp
        config_str = "_".join([
            f"{k}-{v}" for k, v in self.config.items()
        ])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"are_vqa_{config_str}_{timestamp}.json"
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\n✓ Results saved to: {filepath}")


def test_pipeline():
    """Test the complete pipeline on a few samples"""
    logger.info("="*60)
    logger.info("Testing Complete ARE-VQA Pipeline")
    logger.info("="*60)
    
    # Load dataset
    dataset = AOKVQADataset()
    samples = dataset.load_split('val')[:3]  # Test on 3 samples
    
    # Test 1: Full pipeline (use_planner=True, use_knowledge=True)
    logger.info("\n" + "#"*60)
    logger.info("Test 1: Full Pipeline (planner=ON, knowledge=ON)")
    logger.info("#"*60)
    
    pipeline_full = AREVQAPipeline({
        "use_planner": True,
        "use_knowledge": True
    })
    
    results_full = pipeline_full.evaluate(
        samples,
        dataset,
        use_cache=False,
        verbose=True,
        save_results=True
    )
    
    # Test 2: No planner (use_planner=False, use_knowledge=True)
    logger.info("\n" + "#"*60)
    logger.info("Test 2: Ablation - No Planner (planner=OFF, knowledge=ON)")
    logger.info("#"*60)
    
    pipeline_no_planner = AREVQAPipeline({
        "use_planner": False,
        "use_knowledge": True
    })
    
    results_no_planner = pipeline_no_planner.evaluate(
        samples,
        dataset,
        use_cache=True,  # Use cache for faster testing
        verbose=False,
        save_results=True
    )
    
    # Test 3: No knowledge (use_planner=True, use_knowledge=False)
    logger.info("\n" + "#"*60)
    logger.info("Test 3: Ablation - No Knowledge (planner=ON, knowledge=OFF)")
    logger.info("#"*60)
    
    pipeline_no_knowledge = AREVQAPipeline({
        "use_planner": True,
        "use_knowledge": False
    })
    
    results_no_knowledge = pipeline_no_knowledge.evaluate(
        samples,
        dataset,
        use_cache=True,  # Use cache for faster testing
        verbose=False,
        save_results=True
    )
    
    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Full Pipeline:     {results_full['accuracy']:.2%} ({results_full['correct']}/{results_full['total']})")
    logger.info(f"No Planner:        {results_no_planner['accuracy']:.2%} ({results_no_planner['correct']}/{results_no_planner['total']})")
    logger.info(f"No Knowledge:      {results_no_knowledge['accuracy']:.2%} ({results_no_knowledge['correct']}/{results_no_knowledge['total']})")
    logger.info("="*60)


if __name__ == "__main__":
    test_pipeline()
