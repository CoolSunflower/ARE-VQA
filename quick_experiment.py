"""
Quick experiment runner - Compare Baseline vs ARE-VQA Pipeline
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import AOKVQADataset
from src.models.baseline import BaselineVQA
from src.pipeline.are_vqa_pipeline import AREVQAPipeline
from src.utils import setup_logging

logger = setup_logging(__name__)

def main():
    # Load dataset
    dataset = AOKVQADataset()
    samples = dataset.load_split('val')[:20]  # Test on 20 samples
    
    logger.info("="*80)
    logger.info("QUICK EXPERIMENT: Baseline vs ARE-VQA (20 samples)")
    logger.info("="*80)
    
    # Test 1: Baseline
    logger.info("\n" + "#"*80)
    logger.info("TEST 1: BASELINE (Naive VLM)")
    logger.info("#"*80)
    
    baseline = BaselineVQA()
    baseline_correct = 0
    
    for i, sample in enumerate(samples, 1):
        image_path = dataset.get_image_path(sample.split, sample.image_id)
        result = baseline.answer_question(sample, image_path)
        if result['is_correct']:
            baseline_correct += 1
        logger.info(f"[{i}/10] {'✓' if result['is_correct'] else '✗'} {sample.question[:60]}...")
    
    baseline_acc = baseline_correct / len(samples)
    logger.info(f"\nBaseline Accuracy: {baseline_acc:.1%} ({baseline_correct}/{len(samples)})")
    
    # Test 2: Full Pipeline
    logger.info("\n" + "#"*80)
    logger.info("TEST 2: ARE-VQA FULL PIPELINE (Planner + Knowledge)")
    logger.info("#"*80)
    
    pipeline = AREVQAPipeline({"use_planner": True, "use_knowledge": True})
    pipeline_correct = 0
    
    for i, sample in enumerate(samples, 1):
        image_path = dataset.get_image_path(sample.split, sample.image_id)
        result = pipeline.answer_question(sample, image_path, use_cache=True, verbose=False)
        if result['is_correct']:
            pipeline_correct += 1
        logger.info(f"[{i}/10] {'✓' if result['is_correct'] else '✗'} {sample.question[:60]}...")
    
    pipeline_acc = pipeline_correct / len(samples)
    logger.info(f"\nPipeline Accuracy: {pipeline_acc:.1%} ({pipeline_correct}/{len(samples)})")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Baseline:  {baseline_acc:.1%} ({baseline_correct}/10)")
    logger.info(f"Pipeline:  {pipeline_acc:.1%} ({pipeline_correct}/10)")
    logger.info(f"Difference: {(pipeline_acc - baseline_acc):+.1%}")
    logger.info("="*80)
    
    logger.info("\n✓ Quick experiment complete!")
    logger.info("NOTE: This is just 20 samples. Run full experiment for conclusive results.")

if __name__ == "__main__":
    main()
