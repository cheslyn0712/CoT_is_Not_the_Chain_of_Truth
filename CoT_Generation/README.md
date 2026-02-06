# Generation

Generates Chain-of-Thought reasoning data from seed news articles.

## Usage

```bash
python Generation.py
```

## Configuration

Edit `Generation.py`:

- `MODEL_TYPE`: `"qwen3-4b-thinking"`, `"llama3-8b"`, or `"qwen3-vl-8b-thinking"`
- `GPU_IDS`: GPU IDs (e.g., `"0,1"`) or `""` for auto
- `INPUT_FILE`: `"Data/Seed/Real_News.json"`

Edit `prompts_config.json` to define prompt configurations:

```json
{
  "configs": [
    {
      "prompt_type": "indirect",
      "style": "original",
      "prompt": "Your prompt text..."
    }
  ]
}
```

## Input/Output

- **Input**: `Data/Seed/Real_News.json`
- **Output**: `Data/Raw/{model_name}/{prompt_type}/{style}/news.json`
