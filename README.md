# CoT is Not the Chain of Truth

Code repository for "CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation"

## Pipeline Overview

The pipeline consists of three stages:

1. **Generation** (`CoT_Generation/`): Generates CoT reasoning data from seed news articles using different prompt strategies
2. **Annotation** (`CoT_Annotation/`): Labels CoT toxicity and samples data for human verification
3. **Analysis** (`CoT_Analysis/`): Performs layer-level and attention head-level analysis to identify safety-critical components

### Data Flow

```
Seed (Real_News.json) 
  → Raw ({model}/{prompt_type}/{style}/news.json)
    → Processed ({model}/{prompt_type}/{style}/news.json)
      → HumanCheck ({model}/{prompt_type}/{style}/news.json)
```

### Running the Pipeline

1. **Generation**: Fill `CoT_Generation/prompts_config.json` with prompts, then run:
   ```bash
   cd CoT_Generation
   python Generation.py
   ```

2. **Annotation**: Configure `CoT_Annotation/label.py` and run:
   ```bash
   cd CoT_Annotation
   python label.py      # Label toxicity (processes all configurations)
   python verify.py     # Sample one random configuration for human check
   ```

3. **Analysis**: Run layer and head analysis:
   ```bash
   cd CoT_Analysis
   python Layer.py --data_path ... --model_path ... --save_dir ...
   python Head.py --jsons ... --out_root ...
   ```

## Directory Structure

```
CoT_is_Not_the_Chain_of_Truth/
├── CoT_Generation/     # Stage 1: Generate CoT data
├── CoT_Annotation/     # Stage 2: Label and verify
├── CoT_Analysis/       # Stage 3: Analyze attention mechanisms
├── Data/               # All data (Seed → Raw → Processed → HumanCheck)
└── LLM/                # Language models (download separately)
    ├── Qwen3-4B-Thinking-2507/
    ├── Llama-3-8B-Instruct/
    └── Qwen3-VL-8B-Thinking/
```

## Documentation

- **[Data/README.md](Data/README.md)**: Data format specifications
- **[LLM/README.md](LLM/README.md)**: Model download instructions

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

See `requirements.txt` for detailed package versions. Main dependencies:
- Python 3.8+
- PyTorch, Transformers
- NumPy, Matplotlib
- ModelScope (for Qwen3-VL models)

## Setup

### 1. Download Models

Download the required models to `LLM/` directory. See [LLM/README.md](LLM/README.md) for download instructions.

### 2. Prepare Data

Place `Data/Seed/Real_News.json` with seed news articles. Format: `[{"user_input": "...", "toxicity": 0}, ...]`

### 3. Configure Prompts

Fill in `CoT_Generation/prompts_config.json` with your prompt templates.

## Configuration

All scripts use relative paths by default. Configure:
- **Model paths**: Set in each script (default: `LLM/{model_name}`)
- **GPU**: Set `GPU_IDS` in each script (e.g., `"0,1"` or `""` for auto)
- **Data paths**: All relative to project root

See individual README files in each stage directory for detailed configuration instructions.
