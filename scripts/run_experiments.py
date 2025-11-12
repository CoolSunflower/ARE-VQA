"""
Experiment Runner Script
Runs all configurations on the 1000-sample validation subset
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import AOKVQADataset
from src.models.baseline import BaselineVQA
from src.pipeline.are_vqa_pipeline import AREVQAPipeline
from src.utils import setup_logging

logger = setup_logging(__name__)


def run_baseline(dataset, samples, use_cache=True):
    """Run the baseline VLM approach"""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT: Baseline (Naive VLM)")
    logger.info("="*80)
    
    baseline = BaselineVQA()
    results = baseline.evaluate(samples, dataset)
    
    return results


def run_are_vqa(dataset, samples, config, use_cache=True):
    """Run the ARE-VQA pipeline with given configuration"""
    config_name = " + ".join([k for k, v in config.items() if v])
    logger.info("\n" + "="*80)
    logger.info(f"EXPERIMENT: ARE-VQA ({config_name})")
    logger.info("="*80)
    
    pipeline = AREVQAPipeline(config)
    results = pipeline.evaluate(samples, dataset, use_cache=use_cache)
    
    return results


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Run ARE-VQA experiments")
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to evaluate (default: 1000)'
    )
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='Disable model response caching'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['baseline', 'full', 'no_planner', 'no_knowledge', 'minimal'],
        default=['baseline', 'full', 'no_planner', 'no_knowledge'],
        help='Which experiments to run'
    )
    
    args = parser.parse_args()
    use_cache = not args.no_cache
    
    logger.info("="*80)
    logger.info("ARE-VQA EXPERIMENTAL EVALUATION")
    logger.info("="*80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Use cache: {use_cache}")
    logger.info(f"Experiments: {', '.join(args.experiments)}")
    logger.info("="*80)
    
    # Load dataset and samples
    logger.info("\nLoading dataset...")
    dataset = AOKVQADataset()
    
    # Try to load pre-saved subset, otherwise create it
    subset_file = Path("results/validation_subset_1000.json")
    if subset_file.exists():
        logger.info(f"Loading pre-saved subset from {subset_file}")
        import json
        with open(subset_file) as f:
            subset_data = json.load(f)
        samples = [dataset.load_split('val')[i] for i in range(args.num_samples)]
    else:
        logger.info(f"Loading {args.num_samples} samples from validation set")
        all_samples = dataset.load_split('val')
        samples = all_samples[:args.num_samples]
    
    logger.info(f"✓ Loaded {len(samples)} samples")
    
    # Run experiments
    all_results = {}
    
    if 'baseline' in args.experiments:
        all_results['baseline'] = run_baseline(dataset, samples, use_cache)
    
    if 'full' in args.experiments:
        all_results['full_pipeline'] = run_are_vqa(
            dataset,
            samples,
            {"use_planner": True, "use_knowledge": True},
            use_cache
        )
    
    if 'no_planner' in args.experiments:
        all_results['no_planner'] = run_are_vqa(
            dataset,
            samples,
            {"use_planner": False, "use_knowledge": True},
            use_cache
        )
    
    if 'no_knowledge' in args.experiments:
        all_results['no_knowledge'] = run_are_vqa(
            dataset,
            samples,
            {"use_planner": True, "use_knowledge": False},
            use_cache
        )
    
    if 'minimal' in args.experiments:
        all_results['minimal'] = run_are_vqa(
            dataset,
            samples,
            {"use_planner": False, "use_knowledge": False},
            use_cache
        )
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    
    for exp_name, results in all_results.items():
        accuracy = results['accuracy']
        correct = results['correct']
        total = results['total']
        logger.info(f"{exp_name:20s}: {accuracy:.2%} ({correct}/{total})")
    
    logger.info("="*80)
    logger.info("\n✓ All experiments complete!")
    logger.info(f"Results saved to results/ directory")
    logger.info("="*80)


if __name__ == "__main__":
    main()
