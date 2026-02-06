#!/usr/bin/env python3
"""
Human Verification Sampler

This script:
- Reads processed data.
- Filters items where can_generate is false.
- Randomly samples a fixed number of items.
- Writes them to a JSON file for human verification.
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================
# Configuration
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

FIELD_CAN_GENERATE = "can_generate"
SAMPLE_SIZE = 30
RANDOM_SEED = 0

# =========================
# I/O
# =========================

def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def read_items(path: Path) -> List[Dict[str, Any]]:
    if _is_jsonl(path):
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("JSON input must be a JSON array.")
    return obj


def write_json_array(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# Configuration sampling
# =========================

def sample_refusal(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pool = [it for it in items if not bool(it.get(FIELD_CAN_GENERATE, False))]

    random.seed(RANDOM_SEED)
    if len(pool) <= SAMPLE_SIZE:
        return pool[:]
    return random.sample(pool, SAMPLE_SIZE)


def find_all_processed_configs() -> List[Tuple[str, str, str]]:
    """Scan Processed directory and return all available configurations"""
    processed_dir = os.path.join(PROJECT_ROOT, "Data/Processed")
    if not os.path.exists(processed_dir):
        return []
    
    configs = []
    for model_name in os.listdir(processed_dir):
        model_path = os.path.join(processed_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        for prompt_type in os.listdir(model_path):
            prompt_path = os.path.join(model_path, prompt_type)
            if not os.path.isdir(prompt_path):
                continue
            
            for style in os.listdir(prompt_path):
                style_path = os.path.join(prompt_path, style)
                if not os.path.isdir(style_path):
                    continue
                
                news_file = os.path.join(style_path, "news.json")
                if os.path.exists(news_file):
                    configs.append((model_name, prompt_type, style))
    
    return configs


def process_one_config(model_name: str, prompt_type: str, style: str) -> None:
    """Sample items from one configuration for human verification"""
    input_path = os.path.join(PROJECT_ROOT, f"Data/Processed/{model_name}/{prompt_type}/{style}/news.json")
    output_path = os.path.join(PROJECT_ROOT, f"Data/HumanCheck/{model_name}/{prompt_type}/{style}/news.json")

    if not os.path.exists(input_path):
        print(f"⚠️  Skipping {model_name}/{prompt_type}/{style}: input file not found")
        return

    print(f"\nProcessing: {model_name}/{prompt_type}/{style}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    items = read_items(Path(input_path))
    print(f"  Total items: {len(items)}")
    
    sampled = sample_refusal(items)
    print(f"  Sampled {len(sampled)} items (can_generate=false)")
    
    write_json_array(Path(output_path), sampled)
    print(f"  ✓ Saved to: {output_path}")


def main() -> None:
    """Main entry point: randomly selects one configuration and samples items for human verification"""
    print("=" * 80)
    print("CoT VERIFICATION PIPELINE")
    print("=" * 80)

    configs = find_all_processed_configs()
    
    if not configs:
        print("⚠️  No configurations found in Data/Processed/")
        return
    
    print(f"\nFound {len(configs)} configurations")
    
    # Randomly select one configuration
    random.seed(RANDOM_SEED)
    selected_config = random.choice(configs)
    model_name, prompt_type, style = selected_config
    
    print(f"Randomly selected: {model_name}/{prompt_type}/{style}\n")
    
    process_one_config(model_name, prompt_type, style)
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
