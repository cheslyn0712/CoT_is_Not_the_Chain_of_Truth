# Analysis

Mechanistic analysis to identify safety-critical layers and attention heads.

## Layer Analysis (`Layer.py`)

Identifies layers where safe and unsafe reasoning trajectories diverge.

```bash
python Layer.py \
    --data_path Data/Processed/{model}/{prompt_type}/{style}/news.json \
    --model_path LLM/Qwen3-4B-Thinking-2507 \
    --save_dir Data/Safety_Layer/{model}_{prompt_type}_{style}
```

**Output**: `safety_layers.json`, `safety_layers_summary.json`, visualization plots

## Head Analysis (`Head.py`)

Analyzes attention heads within safety-critical layers.

```bash
python Head.py \
    --jsons Data/Processed/Qwen_4B_d_ori.json,Data/Processed/Qwen_4B_i_ori.json \
    --out_root Data/Operator_Analysis \
    --max_n 30 \
    --max_m 30
```

**Output**: For each layer, generates `B1.png`, `B2.png`, `B3.png`, CSV files, and JSON files with critical heads.

**Metrics**:
- B1 (Stability): Spectral norm
- B2 (Geometry): Principal singular vector alignment
- B3 (Energy): Spectral concentration

**Prerequisite**: Run layer analysis first to generate `safety_layers_summary.json`.
