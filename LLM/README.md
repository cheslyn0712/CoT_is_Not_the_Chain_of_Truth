# Models

Download required models to this directory.

## Models

### Qwen3-4B-Thinking-2507

**HuggingFace**: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
```

**ModelScope** (China):
```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507', cache_dir='./Qwen3-4B-Thinking-2507')
```

### Llama-3-8B-Instruct

**HuggingFace**: https://huggingface.co/meta-llama/Llama-3-8B-Instruct

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3-8B-Instruct
```

**Note**: May require access request from Meta.

### Qwen3-VL-8B-Thinking

**ModelScope** (Recommended):
```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Thinking', cache_dir='./Qwen3-VL-8B-Thinking')
```

**HuggingFace**: https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking

**Note**: Requires ModelScope library for loading.

## Directory Structure

After downloading, structure should be:

```
LLM/
├── Qwen3-4B-Thinking-2507/
├── Llama-3-8B-Instruct/
└── Qwen3-VL-8B-Thinking/
```

Model paths are configured in `CoT_Generation/Generation.py` and default to `LLM/{model_name}`.
