# Annotation

Labels CoT toxicity and samples data for human verification.

## Labeling (`label.py`)

Labels toxicity of CoT reasoning using rule-based classification.

**Input**: `Data/Raw/{model_name}/{prompt_type}/{style}/news.json`  
**Output**: `Data/Processed/{model_name}/{prompt_type}/{style}/news.json`

### Configuration

```python
MODEL_NAME = ""      # "" for all, or specific model
PROMPT_TYPE = ""     # "" for all, or specific prompt type
STYLE = ""           # "" for all, or specific style
LLM_ROUTE = ""       # Path to annotation model
GPU_IDS = ""         # GPU IDs or "" for auto
```

### Usage

```bash
python label.py
```

**Logic**: If `can_generate=true`, labels as toxic (`is_cot_toxicity=1`). Otherwise, uses LLM with rules from `Assistant/rules.json`.

## Verification (`verify.py`)

Randomly samples items for human verification.

**Input**: `Data/Processed/{model_name}/{prompt_type}/{style}/news.json`  
**Output**: `Data/HumanCheck/{model_name}/{prompt_type}/{style}/news.json`

### Configuration

```python
SAMPLE_SIZE = 30     # Number of samples
RANDOM_SEED = 0      # Random seed
```

### Usage

```bash
python verify.py
```

Randomly selects one configuration and samples items where `can_generate=false`.
