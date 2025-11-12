"""
Test script to verify dataset setup and structure
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.dataset_loader import AOKVQADataset
from config import RESULTS_DIR, AOKVQA_DIR, COCO_DIR
from utils import setup_logging

logger = setup_logging('dataset_test')


def test_dataset_loading():
    """Test loading the A-OKVQA dataset"""
    logger.info("="*60)
    logger.info("TESTING DATASET LOADING")
    logger.info("="*60)
    
    # Initialize dataset
    logger.info(f"\nInitializing dataset...")
    logger.info(f"A-OKVQA directory: {AOKVQA_DIR}")
    logger.info(f"COCO directory: {COCO_DIR}")
    
    dataset = AOKVQADataset()
    
    # Load validation split
    logger.info(f"\nLoading validation split...")
    val_samples = dataset.load_split('val')
    logger.info(f"✓ Loaded {len(val_samples)} validation samples")
    
    # Display first sample
    logger.info(f"\n{'='*60}")
    logger.info("FIRST SAMPLE DETAILS")
    logger.info(f"{'='*60}")
    sample = val_samples[0]
    logger.info(f"Question ID: {sample.question_id}")
    logger.info(f"Image ID: {sample.image_id}")
    logger.info(f"Question: {sample.question}")
    logger.info(f"Choices:")
    for i, choice in enumerate(sample.choices):
        marker = "✓" if i == sample.correct_choice_idx else " "
        logger.info(f"  {chr(65+i)}) {choice} {marker}")
    logger.info(f"Correct Answer: {sample.correct_choice_letter}) {sample.correct_choice}")
    if sample.rationales:
        logger.info(f"Rationale: {sample.rationales[0]}")
    
    # Get statistics
    logger.info(f"\n{'='*60}")
    logger.info("DATASET STATISTICS")
    logger.info(f"{'='*60}")
    stats = dataset.get_statistics(val_samples)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Create and save subset
    logger.info(f"\n{'='*60}")
    logger.info("CREATING SUBSET")
    logger.info(f"{'='*60}")
    subset_file = RESULTS_DIR / "validation_subset_1000.json"
    subset = dataset.create_subset('val', size=1000, save_to=subset_file)
    logger.info(f"✓ Created subset of {len(subset)} samples")
    logger.info(f"✓ Saved subset info to: {subset_file}")
    
    # Test loading subset from file
    logger.info(f"\nTesting subset loading from file...")
    loaded_subset = dataset.load_subset_from_file(subset_file)
    logger.info(f"✓ Loaded {len(loaded_subset)} samples from file")
    
    # Verify they match
    assert len(subset) == len(loaded_subset), "Subset sizes don't match!"
    assert all(s1.question_id == s2.question_id for s1, s2 in zip(subset, loaded_subset)), \
        "Subset order doesn't match!"
    logger.info(f"✓ Subset verification passed")
    
    # Check image paths (note: we may not have COCO images downloaded)
    logger.info(f"\n{'='*60}")
    logger.info("CHECKING IMAGE PATHS")
    logger.info(f"{'='*60}")
    sample = val_samples[0]
    image_path = dataset.get_image_path(sample.split, sample.image_id)
    logger.info(f"Sample image path: {image_path}")
    
    if image_path.exists():
        logger.info(f"✓ Image file exists")
        logger.info(f"  Size: {image_path.stat().st_size / 1024:.1f} KB")
    else:
        logger.warning(f"⚠ Image file not found")
        logger.warning(f"  COCO images need to be downloaded separately")
        logger.warning(f"  This is expected if you haven't downloaded COCO dataset")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ ALL DATASET TESTS PASSED")
    logger.info(f"{'='*60}\n")
    
    return True


def test_dataset_structure():
    """Verify the expected structure of dataset samples"""
    logger.info("="*60)
    logger.info("TESTING DATASET STRUCTURE")
    logger.info("="*60)
    
    dataset = AOKVQADataset()
    samples = dataset.load_split('val')
    
    # Check sample structure
    sample = samples[0]
    
    required_fields = [
        'question_id', 'image_id', 'question', 'choices',
        'correct_choice_idx', 'split'
    ]
    
    logger.info(f"\nChecking required fields...")
    for field in required_fields:
        assert hasattr(sample, field), f"Missing field: {field}"
        logger.info(f"✓ {field}: {type(getattr(sample, field))}")
    
    # Check data types
    logger.info(f"\nChecking data types...")
    assert isinstance(sample.question_id, str), "question_id should be string"
    assert isinstance(sample.image_id, int), "image_id should be int"
    assert isinstance(sample.question, str), "question should be string"
    assert isinstance(sample.choices, list), "choices should be list"
    assert len(sample.choices) == 4, "Should have 4 choices"
    assert isinstance(sample.correct_choice_idx, int), "correct_choice_idx should be int"
    assert 0 <= sample.correct_choice_idx < 4, "correct_choice_idx should be 0-3"
    logger.info(f"✓ All data types are correct")
    
    # Check properties
    logger.info(f"\nChecking computed properties...")
    assert sample.correct_choice == sample.choices[sample.correct_choice_idx]
    logger.info(f"✓ correct_choice: {sample.correct_choice}")
    assert sample.correct_choice_letter in ['A', 'B', 'C', 'D']
    logger.info(f"✓ correct_choice_letter: {sample.correct_choice_letter}")
    
    logger.info(f"\n✓ DATASET STRUCTURE TESTS PASSED\n")
    
    return True


if __name__ == "__main__":
    try:
        test_dataset_structure()
        test_dataset_loading()
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        logger.info("="*60 + "\n")
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
