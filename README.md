# ARE-VQA: Adaptive Reasoning Engine for Visual Question Answering

A modular five-stage reasoning pipeline for knowledge-based Visual Question Answering, evaluated on the A-OKVQA dataset using local vision-language models.

## Documentation

- **[Report-ARE-VQA.pdf](Report-ARE-VQA.pdf)**: Complete research report with methodology, experiments, and results
- **[Presentation-Deck-ARE-VQA.pdf](Presentation-Deck-ARE-VQA.pdf)**: Presentation slides with visualizations and key findings

## Overview

ARE-VQA addresses the challenge of answering visual questions that require external knowledge by decomposing the reasoning process into five specialized modules:

1. **Triage Router**: Classifies questions by complexity and knowledge requirements
2. **Context Builder**: Extracts visual context and retrieves external knowledge using LLM
3. **Query Planner**: Decomposes complex questions into sub-questions
4. **Tool Executor**: Answers sub-questions using vision model
5. **Synthesizer**: Combines intermediate answers into final response

**Key Innovation**: Uses LLM as a knowledge agent (along with APIs) enabling fully local deployment with no external dependencies. Fully agentic pipeline.

## Dataset

**A-OKVQA (Augmented OK-VQA)**
- Crowdsourced dataset requiring commonsense and world knowledge
- 25K questions designed to be unanswerable by knowledge bases alone
- Validation set: 1,145 samples
- Multiple choice format (4 options per question)
- Includes human-written rationales

## Setup Instructions

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- ~15GB disk space for datasets
- ~13GB for models (llama3.2-vision + qwen3:8b)

### 1. Install Dependencies

```bash
pip install ollama pillow
```

### 2. Download Models

```bash
# Vision model for image understanding
ollama pull llama3.2-vision

# Text model for reasoning and knowledge
ollama pull qwen3:8b
```

### 3. Download A-OKVQA Dataset

```bash
export AOKVQA_DIR=./datasets/aokvqa/
mkdir -p ${AOKVQA_DIR}

curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}
```

### 4. Download COCO Images

```bash
export COCO_DIR=./datasets/coco/
mkdir -p ${COCO_DIR}

# Download train, val, and test image sets
for split in train val test; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}
    rm "${split}2017.zip"
done

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}
rm annotations_trainval2017.zip
```

## Running Experiments

### Quick Test (20 samples)

```bash
python quick_experiment.py
```

### Full Evaluation

```bash
# Run experiments with different configurations
python scripts/run_experiments.py --config full_pipeline
python scripts/run_experiments.py --config no_planner
python scripts/run_experiments.py --config no_knowledge
python scripts/run_experiments.py --config baseline
```

### Configuration

Edit `src/config.py` to customize:
- Model selection and parameters
- Pipeline module toggles
- Temperature settings
- Cache and logging paths
- Dataset subset size

## Project Structure

```
├── src/
│   ├── modules/          # Five pipeline modules
│   ├── models/           # Model wrappers (Ollama)
│   ├── prompts/          # Prompt templates
│   ├── data/             # Dataset loaders
│   └── pipeline/         # Pipeline orchestration
├── datasets/
│   ├── aokvqa/          # A-OKVQA annotations
│   └── coco/            # COCO images
├── scripts/             # Experiment runners
├── results/             # Evaluation outputs
├── Report-ARE-VQA.pdf           # Detailed research report
└── Presentation-Deck-ARE-VQA.pdf            # Presentation slides
```

## Key Features

- **Fully Local**: No external API dependencies (Wikipedia, Google, etc.)
- **LLM Knowledge Agent**: Uses Qwen3:8b for knowledge retrieval
- **Modular Design**: Each stage can be ablated for analysis
- **Token Limits**: Prevents infinite loops and massive context
- **Caching**: Speeds up repeated evaluations
- **Transparent Traces**: Full reasoning logs for qualitative analysis

## Results Summary

See **[Report-ARE-VQA.pdf](Report-ARE-VQA.pdf)** for complete results and analysis.

**Key Findings**:
- Pipeline demonstrates effectiveness on compositional and knowledge-based questions
- LLM knowledge agent successfully replaces external API dependencies
- Modular design enables systematic ablation studies
- Token limits critical for preventing infinite loops in vision models

## Citation

If you use this code or methodology, please reference:

```
ARE-VQA: Agentic Reasoning Engine for Visual Question Answering
Adarsh Gupta, Abhishek Kumar
Indian Institute of Technology Guwahati, 2025
```

## Contact

- Adarsh Gupta - [adarsh.gupta@iitg.ac.in]

Indian Institute of Technology Guwahati

## License

This project is for academic research purposes.
