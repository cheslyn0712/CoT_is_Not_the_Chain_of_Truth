# Data Directory Structure

This directory contains all data files for the pipeline. 

## Directory Structure

```
Data/
├── Seed/
│   └── seed.json                    # Input: All seed news articles
├── Raw/                             # Generated CoT data (before annotation)
│   ├── {model_name}/
│   │   ├── {prompt_type}/
│   │   │   ├── {style}/
│   │   │   │   └── news.json
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── Processed/                       # Processed data with labels
│   ├── {model_name}/
│   │   ├── {prompt_type}/
│   │   │   ├── {style}/
│   │   │   │   └── news.json
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── HumanCheck/                     # Samples for human verification
    ├── {model_name}/
    │   ├── {prompt_type}/
    │   │   ├── {style}/
    │   │   │   └── news.json
    │   │   └── ...
    │   └── ...
    └── ...
```

## Model Names

- `Qwen_4B`: Qwen3-4B-Thinking model
- `Llama_8B`: Llama-3-8B-Instruct model
- `Qwen_8B`: Qwen3-VL-8B-Thinking model

## Prompt Types

- `direct`: Direct prompt strategy (explicitly asks model to generate fake news)
- `indirect`: Indirect prompt strategy (disguised as creative writing task)

## Styles

- `original`: Original style (no style modification)
- `BBC`: BBC news style
- `NY`: New York Times style
- (Other styles as configured in `prompts_config.json`)

## Data Format

### Seed (`seed.json`)

Input file for the Generation stage. Contains an array of seed news articles:

```json
[
  {
    "user_input": "News article text here..."
  },
  {
    "user_input": "Another news article..."
  },
  ...
]
```

**Field Description**:
- `user_input` (string): The original news article text that will be used as input for fake news generation

**Example**:
```json
[
  {
    "user_input": "Scientists have discovered a new species of butterfly in the Amazon rainforest. The discovery was made during a research expedition..."
  }
]
```

### Raw (`{model_name}/{prompt_type}/{style}/news.json`)

Generated CoT data from the Generation stage. Contains an array of processed items with Chain-of-Thought reasoning:

```json
[
  {
    "id": 0,
    "is_direct_fake_prompt": 0,
    "prompt_strategy": "Prompt strategy text (without OUTPUT FORMAT section)...",
    "input_news": "Original news text from seed...",
    "input_news_labels": 0,
    "is_cot_toxicity": 1,
    "can_generate": true,
    "out_news": "Generated fake news content (if can_generate=true)...",
    "CoT": "Full Chain-of-Thought reasoning process...",
    "output_news_labels": 1
  },
  ...
]
```

**Field Descriptions**:
- `id` (int): Unique identifier for the news item (matches index from seed)
- `is_direct_fake_prompt` (int): `1` if direct prompt was used, `0` if indirect prompt
- `prompt_strategy` (string): The prompt template used (OUTPUT FORMAT section removed)
- `input_news` (string): Original news article from seed data
- `input_news_labels` (int): Label for input news (always `0` for seed data)
- `is_cot_toxicity` (int): Default toxicity label for CoT (`1` by default)
- `can_generate` (bool): Whether the model decided it can generate fake news
  - `true`: Model generated fake news content
  - `false`: Model refused to generate (safety mechanism triggered)
- `out_news` (string): Generated fake news content (empty string if `can_generate=false`)
- `CoT` (string): Complete Chain-of-Thought reasoning process, including internal deliberation
- `output_news_labels` (int): Label for output (`1` if `can_generate=true`, `0` otherwise)

**Important Notes**:
- When `can_generate=false`, the `CoT` field contains the model's refusal reasoning
- The `CoT` field contains the full reasoning process, which may include unsafe content even when the model refuses

### Processed (`{model_name}/{prompt_type}/{style}/news.json`)

Annotated data from the Annotation stage. Same structure as Raw, with updated toxicity labels:

```json
[
  {
    "id": 0,
    "is_direct_fake_prompt": 0,
    "prompt_strategy": "Prompt strategy text...",
    "input_news": "Original news text...",
    "input_news_labels": 0,
    "is_cot_toxicity": 1,  // Updated by annotation model
    "can_generate": true,
    "out_news": "Generated fake news...",
    "CoT": "Chain-of-Thought reasoning...",
    "output_news_labels": 1
  },
  ...
]
```

**Field Descriptions**:
- All fields same as Raw, except:
- `is_cot_toxicity` (int): **Updated toxicity label** for the CoT reasoning
  - `1`: CoT contains toxic content (violates rules)
  - `0`: CoT is safe (does not violate rules)
  - If `can_generate=true`, automatically labeled as `1`
  - If `can_generate=false`, determined by LLM annotation using rules from `Assistant/rules.json`

### HumanCheck (`{model_name}/{prompt_type}/{style}/news.json`)

Sampled data for human verification. Same format as Processed, but contains only sampled items:

```json
[
  {
    "id": 5,
    "is_direct_fake_prompt": 0,
    "prompt_strategy": "...",
    "input_news": "...",
    "input_news_labels": 0,
    "is_cot_toxicity": 1,
    "can_generate": false,  // Only items with can_generate=false are sampled
    "out_news": "",
    "CoT": "Model's refusal reasoning...",
    "output_news_labels": 0
  },
  ...
]
```

**Characteristics**:
- Contains only items where `can_generate=false` (model refused to generate)
- Default sample size: 30 items
- Randomly selected from one randomly chosen configuration
- Used for human verification of toxicity labels

## Data Flow

1. **Seed** (`seed.json`): Single file containing seed news articles
   - Input to Generation stage
   - Format: `[{user_input: "..."}, ...]`

2. **Raw** (`{model}/{prompt_type}/{style}/news.json`): Generated CoT data
   - Output from Generation stage
   - Input to Annotation stage
   - Contains full CoT reasoning and generation results
   - Format: `[{id, is_direct_fake_prompt, prompt_strategy, input_news, ..., CoT, can_generate, ...}, ...]`

3. **Processed** (`{model}/{prompt_type}/{style}/news.json`): Annotated data
   - Output from Annotation stage
   - Input to Analysis/Verification stage
   - Same structure as Raw, with updated `is_cot_toxicity` labels
   - Format: Same as Raw, with toxicity labels updated

4. **HumanCheck** (`{model}/{prompt_type}/{style}/news.json`): Sampled verification data
   - Output from Verification stage
   - Contains only refused items (`can_generate=false`)
   - Format: Same as Processed, subset of items

## Notes

- All directories are created automatically by the pipeline
- File names are always `news.json` (consistent across all stages)
- Directory structure is maintained from Raw onwards: `{model_name}/{prompt_type}/{style}/`
- Seed is a single file; all other stages use the hierarchical structure
- The pipeline supports multiple models, prompt types, and styles simultaneously
- Each configuration combination generates a separate `news.json` file

## Generating Your Own Data

To generate data:

1. **Prepare Seed Data**: Create `Data/Seed/seed.json` with your news articles
2. **Configure Prompts**: Fill in `CoT_Generation/prompts_config.json` with your prompts
3. **Run Generation**: Execute `CoT_Generation/Generation.py` to generate Raw data
4. **Run Annotation**: Execute `CoT_Annotation/label.py` to label toxicity
5. **Run Verification**: Execute `CoT_Annotation/verify.py` to sample for human check

See the main README and individual stage READMEs for detailed instructions.