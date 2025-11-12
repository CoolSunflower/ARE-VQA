"""
Configuration file for ARE-VQA Pipeline
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
AOKVQA_DIR = DATASETS_DIR / "aokvqa"
COCO_DIR = DATASETS_DIR / "coco"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model configurations
OLLAMA_VISION_MODEL = "llama3.2-vision"
OLLAMA_LLM_MODEL = "qwen3:8b"  # Using qwen3:8b for reasoning

# Pipeline configurations
PIPELINE_CONFIG = {
    "use_planner": True,
    "use_knowledge": True,
}

ABLATION_CONFIGS = {
    "no_planner": {
        "use_planner": False,
        "use_knowledge": True,
    },
    "no_knowledge": {
        "use_planner": True,
        "use_knowledge": False,
    },
    "full_pipeline": {
        "use_planner": True,
        "use_knowledge": True,
    }
}

# Dataset configurations
SUBSET_SIZE = 1000  # Number of samples to use from validation set
RANDOM_SEED = 42

# Retrieval configurations
RETRIEVAL_CONFIG = {
    "num_results": 5,
    "max_snippet_length": 500,
}

# Temperature settings for different modules
TEMPERATURE_CONFIG = {
    "triage": 0.0,      # Deterministic for classification
    "planner": 0.0,     # Deterministic for planning
    "executor": 0.0,    # Deterministic for answering
    "synthesizer": 0.0, # Deterministic for final answer
    "query_gen": 0.3,   # Slightly creative for query generation
}

# Logging configuration
LOG_LEVEL = "INFO"
ENABLE_CACHING = True
CACHE_RESPONSES = True

# Evaluation settings
EVAL_BATCH_SIZE = 1  # Process one question at a time for now
SAVE_INTERMEDIATE_RESULTS = True
SAVE_TRACES = True  # Save full pipeline traces for qualitative analysis
