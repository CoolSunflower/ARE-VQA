"""
Dataset loader for A-OKVQA dataset
Handles loading, subsetting, and managing the dataset
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import AOKVQA_DIR, COCO_DIR, SUBSET_SIZE, RANDOM_SEED
from utils import setup_logging, save_json, load_json

logger = setup_logging(__name__)


@dataclass
class AOKVQASample:
    """Data class for a single A-OKVQA sample"""
    question_id: str
    image_id: int
    question: str
    choices: List[str]
    correct_choice_idx: int
    split: str
    direct_answers: List[str] = None
    rationales: List[str] = None
    difficult_direct_answer: bool = False
    
    @property
    def correct_choice(self) -> str:
        """Get the correct choice text"""
        return self.choices[self.correct_choice_idx]
    
    @property
    def correct_choice_letter(self) -> str:
        """Get the correct choice as a letter (A, B, C, D)"""
        letters = ['A', 'B', 'C', 'D']
        return letters[self.correct_choice_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'question_id': self.question_id,
            'image_id': self.image_id,
            'question': self.question,
            'choices': self.choices,
            'correct_choice_idx': self.correct_choice_idx,
            'correct_choice': self.correct_choice,
            'correct_choice_letter': self.correct_choice_letter,
            'split': self.split,
            'direct_answers': self.direct_answers,
            'rationales': self.rationales,
            'difficult_direct_answer': self.difficult_direct_answer,
        }


class AOKVQADataset:
    """A-OKVQA Dataset Manager"""
    
    def __init__(
        self, 
        aokvqa_dir: Path = AOKVQA_DIR,
        coco_dir: Path = COCO_DIR,
        version: str = 'v1p0'
    ):
        """
        Initialize the A-OKVQA dataset manager
        
        Args:
            aokvqa_dir: Directory containing A-OKVQA JSON files
            coco_dir: Directory containing COCO images
            version: Dataset version (default: v1p0)
        """
        self.aokvqa_dir = Path(aokvqa_dir)
        self.coco_dir = Path(coco_dir)
        self.version = version
        
        logger.info(f"Initialized A-OKVQA dataset manager")
        logger.info(f"A-OKVQA directory: {self.aokvqa_dir}")
        logger.info(f"COCO directory: {self.coco_dir}")
    
    def load_split(self, split: str) -> List[AOKVQASample]:
        """
        Load a specific split of the dataset
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            List of AOKVQASample objects
        """
        assert split in ['train', 'val', 'test'], \
            f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
        
        filepath = self.aokvqa_dir / f"aokvqa_{self.version}_{split}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}\n"
                f"Please download the A-OKVQA dataset to {self.aokvqa_dir}"
            )
        
        logger.info(f"Loading {split} split from {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = [
            AOKVQASample(
                question_id=item['question_id'],
                image_id=item['image_id'],
                question=item['question'],
                choices=item['choices'],
                correct_choice_idx=item['correct_choice_idx'],
                split=item['split'],
                direct_answers=item.get('direct_answers'),
                rationales=item.get('rationales'),
                difficult_direct_answer=item.get('difficult_direct_answer', False),
            )
            for item in data
        ]
        
        logger.info(f"Loaded {len(samples)} samples from {split} split")
        return samples
    
    def get_image_path(self, split: str, image_id: int) -> Path:
        """
        Get the path to a COCO image
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            image_id: COCO image ID
            
        Returns:
            Path to the image file
        """
        # COCO images are in split2017 directories
        image_dir = self.coco_dir / f"{split}2017"
        image_path = image_dir / f"{image_id:012d}.jpg"
        
        return image_path
    
    def create_subset(
        self, 
        split: str, 
        size: int = SUBSET_SIZE, 
        random_seed: int = RANDOM_SEED,
        save_to: Optional[Path] = None
    ) -> List[AOKVQASample]:
        """
        Create a random subset of the dataset
        
        Args:
            split: Dataset split to sample from
            size: Number of samples to include
            random_seed: Random seed for reproducibility
            save_to: Optional path to save the subset indices
            
        Returns:
            List of sampled AOKVQASample objects
        """
        all_samples = self.load_split(split)
        
        if size >= len(all_samples):
            logger.warning(
                f"Requested subset size ({size}) >= total samples ({len(all_samples)}). "
                f"Returning all samples."
            )
            return all_samples
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        subset = random.sample(all_samples, size)
        
        logger.info(
            f"Created subset of {size} samples from {split} split "
            f"(seed={random_seed})"
        )
        
        # Save subset question IDs for reproducibility
        if save_to:
            subset_info = {
                'split': split,
                'size': size,
                'random_seed': random_seed,
                'question_ids': [s.question_id for s in subset],
                'image_ids': [s.image_id for s in subset],
            }
            save_json(subset_info, save_to)
            logger.info(f"Saved subset information to {save_to}")
        
        return subset
    
    def load_subset_from_file(self, subset_file: Path) -> List[AOKVQASample]:
        """
        Load a previously saved subset
        
        Args:
            subset_file: Path to the subset information JSON file
            
        Returns:
            List of AOKVQASample objects
        """
        subset_info = load_json(subset_file)
        all_samples = self.load_split(subset_info['split'])
        
        # Create a mapping from question_id to sample
        id_to_sample = {s.question_id: s for s in all_samples}
        
        # Reconstruct the subset in the same order
        subset = [
            id_to_sample[qid] 
            for qid in subset_info['question_ids']
            if qid in id_to_sample
        ]
        
        logger.info(
            f"Loaded subset of {len(subset)} samples from {subset_file}"
        )
        
        return subset
    
    def get_statistics(self, samples: List[AOKVQASample]) -> Dict[str, Any]:
        """
        Get statistics about a list of samples
        
        Args:
            samples: List of AOKVQASample objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(samples),
            'avg_question_length': sum(len(s.question.split()) for s in samples) / len(samples),
            'avg_choices': sum(len(s.choices) for s in samples) / len(samples),
            'difficult_answers': sum(1 for s in samples if s.difficult_direct_answer),
        }
        
        return stats


def main():
    """Test the dataset loader"""
    logger.info("Testing A-OKVQA Dataset Loader")
    
    # Initialize dataset
    dataset = AOKVQADataset()
    
    # Load validation split
    val_samples = dataset.load_split('val')
    logger.info(f"Validation set size: {len(val_samples)}")
    
    # Show first sample
    sample = val_samples[0]
    logger.info(f"\nFirst sample:")
    logger.info(f"  Question ID: {sample.question_id}")
    logger.info(f"  Image ID: {sample.image_id}")
    logger.info(f"  Question: {sample.question}")
    logger.info(f"  Choices: {sample.choices}")
    logger.info(f"  Correct: {sample.correct_choice} ({sample.correct_choice_letter})")
    
    # Create a subset
    from config import RESULTS_DIR
    subset_file = RESULTS_DIR / "validation_subset_1000.json"
    subset = dataset.create_subset('val', size=1000, save_to=subset_file)
    
    # Get statistics
    stats = dataset.get_statistics(subset)
    logger.info(f"\nSubset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nDataset loader test completed successfully!")


if __name__ == "__main__":
    main()
