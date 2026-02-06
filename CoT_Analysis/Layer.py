#!/usr/bin/env python3
"""
CoT Analysis Pipeline - Layer-level Analysis

This script performs layer-level analysis on processed CoT data.
"""

import os
import json
import pickle
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Get script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ===============================
# Global plotting style (Times New Roman, compact, uniform)
# ===============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# ===============================
# Qwen3-VL support (modelscope)
# ===============================
try:
    from modelscope import Qwen3VLForConditionalGeneration
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False

warnings.filterwarnings("ignore")


# ============================================================
# JSON loader
# ============================================================

def _get_nested(d: Dict[str, Any], path: str):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_cot_json(
    json_path: str,
    cot_field: str = "CoT",
    toxicity_field: str = "is_cot_toxicity",
    toxic_value: int = 1,
) -> Tuple[List[str], List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normal, malicious = [], []
    for item in data:
        cot = item.get(cot_field, "")
        lab = _get_nested(item, toxicity_field)

        if not isinstance(cot, str) or not cot.strip():
            continue
        if lab is None:
            continue

        if int(lab) == int(toxic_value):
            malicious.append(cot.strip())
        else:
            normal.append(cot.strip())

    return normal, malicious


# ============================================================
# Model loading (text-only; avoid lm_head)
# ============================================================

def _looks_like_qwen3_vl(model_path: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        mt = getattr(cfg, "model_type", "")
        cls = cfg.__class__.__name__.lower()
        return ("qwen3_vl" in str(mt).lower()) or ("qwen3vl" in cls) or ("qwen3_vl" in cls)
    except Exception:
        return "qwen3-vl" in model_path.lower() or "qwen3_vl" in model_path.lower()


def _is_qwen_family(model_path: str) -> bool:
    return "qwen" in str(model_path).lower()


def _is_llama_family(model_path: str) -> bool:
    s = str(model_path).lower()
    return ("llama" in s) or ("llama-" in s) or ("llama_" in s)


def load_text_only_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_vl = _looks_like_qwen3_vl(model_path)

    if is_vl:
        if not HAS_QWEN3VL:
            raise RuntimeError(
                "Detected Qwen3-VL model, but modelscope is not installed.\n"
                "Please `pip install modelscope` in your env."
            )
        print("[INFO] Loading Qwen3-VL via modelscope (text-only forward)")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError("Qwen3VLForConditionalGeneration has no `.model` attribute; please check modelscope version.")
    else:
        print("[INFO] Loading standard CausalLM via transformers")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError("AutoModelForCausalLM has no `.model` attribute for this checkpoint.")

    model.eval()
    backbone.eval()
    return model, backbone, tokenizer


# ============================================================
# Hidden state extraction
# ============================================================

@torch.no_grad()
def get_last_token_vectors_all_layers(
    backbone,
    tokenizer,
    text: str,
    max_length: int = 512,
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    device = next(backbone.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = backbone(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        output_hidden_states=True,
        use_cache=False,
    )

    hs = outputs.hidden_states
    vecs = [hs[i][0, -1].detach().float().cpu().numpy() for i in range(len(hs))]
    return vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


# ============================================================
# r-pair cosine sampling
# ============================================================

def get_r_lists_cossim(
    backbone,
    tokenizer,
    sentences_1: List[str],
    sentences_2: List[str],
    seed: int,
    r: int,
    max_length: int,
    allow_same_within_class: bool = True,
):
    rng = random.Random(seed)
    allcos: List[List[float]] = []

    if len(sentences_1) == 0 or len(sentences_2) == 0:
        return allcos

    same_pool = (sentences_1 is sentences_2)

    for _ in range(r):
        if same_pool:
            if len(sentences_1) >= 2:
                a, b = rng.sample(sentences_1, 2)
            else:
                if not allow_same_within_class:
                    break
                a = sentences_1[0]
                b = sentences_1[0]
        else:
            a = rng.choice(sentences_1)
            b = rng.choice(sentences_2)

        v1 = get_last_token_vectors_all_layers(backbone, tokenizer, a, max_length=max_length)
        v2 = get_last_token_vectors_all_layers(backbone, tokenizer, b, max_length=max_length)

        L = min(len(v1), len(v2))
        cso = [cosine_similarity(v1[i], v2[i]) for i in range(1, L)]  # skip emb
        allcos.append(cso)

    return allcos


# ============================================================
# Qwen: remove last layer everywhere (stats / scaling / localization / plots)
# ============================================================

def _trim_last_layer(allcos: List[List[float]]) -> List[List[float]]:
    trimmed = []
    for row in allcos:
        if row is None:
            continue
        trimmed.append(row[:-1] if len(row) >= 1 else row)
    return trimmed


# ============================================================
# Layer analysis
# ============================================================

def build_layer_analysis(allcos_NN, allcos_NM):
    cos_NN = np.array(allcos_NN)
    cos_NM = np.array(allcos_NM)

    mean_cos_NN = np.mean(cos_NN, axis=0)
    mean_cos_NM = np.mean(cos_NM, axis=0)
    std_cos_NN = np.std(cos_NN, axis=0)
    std_cos_NM = np.std(cos_NM, axis=0)

    angles_NN = np.arccos(np.clip(cos_NN, -1.0, 1.0))
    angles_NM = np.arccos(np.clip(cos_NM, -1.0, 1.0))

    angle_diff = angles_NM - angles_NN
    mean_angle_diff = np.mean(angle_diff, axis=0)
    std_angle_diff = np.std(angle_diff, axis=0)

    cos_gap = mean_cos_NN - mean_cos_NM
    angle_diff_gradient = np.gradient(mean_angle_diff)
    cos_gap_gradient = np.gradient(cos_gap)

    layer_analysis = {
        "layer_indices": list(range(len(mean_angle_diff))),
        "mean_cos_NN": mean_cos_NN.tolist(),
        "mean_cos_NM": mean_cos_NM.tolist(),
        "std_cos_NN": std_cos_NN.tolist(),
        "std_cos_NM": std_cos_NM.tolist(),
        "cos_gap": cos_gap.tolist(),
        "cos_gap_gradient": cos_gap_gradient.tolist(),
        "mean_angle_diff": mean_angle_diff.tolist(),
        "std_angle_diff": std_angle_diff.tolist(),
        "angle_diff_gradient": angle_diff_gradient.tolist(),
        "mean_angle_NN": np.mean(angles_NN, axis=0).tolist(),
        "mean_angle_NM": np.mean(angles_NM, axis=0).tolist(),
    }

    return layer_analysis, mean_angle_diff, mean_cos_NN, mean_cos_NM


# ============================================================
# best window (fixed w=3)
# ============================================================

def best_window_for_w(signal: np.ndarray, w: int):
    signal = np.asarray(signal, dtype=np.float64)
    L = len(signal)
    if L <= 0:
        raise RuntimeError("signal length is 0, cannot localize window.")
    w = min(w, L)
    best_score = -1e18
    best_s = 0
    for s in range(0, L - w + 1):
        sc = float(np.mean(signal[s:s + w]))
        if sc > best_score:
            best_score = sc
            best_s = s
    return best_score, best_s, best_s + w - 1


# ============================================================
# Global math scaling
# ============================================================

def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


def global_adaptive_power_scale_signed(x: np.ndarray, q: float = 0.95, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    t = np.abs(x)

    m = float(np.max(t)) if t.size > 0 else 0.0
    if not np.isfinite(m) or m < 1e-30:
        return x.copy(), 1.0

    p = _safe_quantile(t, q)
    r = float(p / (m + eps))

    if not np.isfinite(r) or r <= 0.0:
        gamma = 1.0
    elif r >= 1.0 - 1e-6:
        gamma = 1.0
    else:
        gamma = float(np.log(0.5) / np.log(r))
        if gamma < 1.0 or not np.isfinite(gamma):
            gamma = 1.0

    u = t / (m + eps)
    y = np.sign(x) * (u ** gamma) * m
    return y.astype(np.float32), gamma


def compute_global_math_scaled_curves(
    mean_cos_NN: np.ndarray,
    std_cos_NN: np.ndarray,
    mean_cos_NM: np.ndarray,
    std_cos_NM: np.ndarray,
    q: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mean_cos_NN = np.asarray(mean_cos_NN, dtype=np.float32)
    mean_cos_NM = np.asarray(mean_cos_NM, dtype=np.float32)
    std_cos_NN  = np.asarray(std_cos_NN,  dtype=np.float32)
    std_cos_NM  = np.asarray(std_cos_NM,  dtype=np.float32)

    mid = (mean_cos_NN + mean_cos_NM) / 2.0
    gap = (mean_cos_NN - mean_cos_NM)

    gap2, gamma = global_adaptive_power_scale_signed(gap, q=q)

    nn2 = mid + gap2 / 2.0
    nm2 = mid - gap2 / 2.0

    gmax = float(np.max(np.abs(gap))) + 1e-12
    u = np.abs(gap) / gmax
    std_scale = (u ** max(0.0, (gamma - 1.0))).astype(np.float32)

    nn_std2 = std_cos_NN * std_scale
    nm_std2 = std_cos_NM * std_scale

    nn2 = np.clip(nn2, 0.0, 1.0)
    nm2 = np.clip(nm2, 0.0, 1.0)

    gap2 = nn2 - nm2
    return nn2, nn_std2, nm2, nm_std2, gap2, float(gamma)


# ============================================================
# Plotting helpers 
# ============================================================

def _layer_percent_axis(L: int) -> np.ndarray:
    if L <= 1:
        return np.array([100.0], dtype=np.float64)
    return (np.arange(L, dtype=np.float64) / (L - 1)) * 100.0


def _percent_step(L: int) -> float:
    if L <= 1:
        return 100.0
    return 100.0 / (L - 1)


def _inner_ticks(ymin: float, ymax: float, n: int) -> List[float]:
    # exclude both ends, take n ticks in the middle
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin + 1e-12:
        return [float(ymin)]
    ticks = np.linspace(ymin, ymax, n + 2)[1:-1]
    out = []
    for t in ticks:
        t = float(t)
        if len(out) == 0 or abs(t - out[-1]) > 1e-10:
            out.append(t)
    return out


def plot_two_panel_cosine_and_gap(
    mean_cos_NN: np.ndarray,
    std_cos_NN: np.ndarray,
    mean_cos_NM: np.ndarray,
    std_cos_NM: np.ndarray,
    safety_start: int,
    safety_end: int,
    save_path: str,
    *,
    gap_override: Optional[np.ndarray] = None,
    legend_loc: str = "best",
):
    """
    Compact & paper-friendly:
      - panels almost touching
      - robust tight y-range
      - x axis uses percent-depth coords but xlabel is "Layer"
      - avoid clipping with tiny x padding + slightly larger left/right margins
      - darker Gap bars
      - legend loc is controllable (Llama: lower right; Qwen: upper left)
    """
    mean_cos_NN = np.asarray(mean_cos_NN, dtype=np.float32)
    mean_cos_NM = np.asarray(mean_cos_NM, dtype=np.float32)
    std_cos_NN  = np.asarray(std_cos_NN,  dtype=np.float32)
    std_cos_NM  = np.asarray(std_cos_NM,  dtype=np.float32)

    L = len(mean_cos_NN)
    x = _layer_percent_axis(L)
    step = _percent_step(L)

    gap = (mean_cos_NN - mean_cos_NM) if gap_override is None else np.asarray(gap_override, dtype=np.float32)

    # ---- compact geometry ----
    FIG_W, FIG_H = 4.8, 3.2
    LINE_W = 1.9
    FILL_ALPHA = 0.14
    GRID_ALPHA = 0.14
    SPAN_ALPHA = 0.14

    # Darker + more opaque gap bars (visibly darker than before)
    BAR_ALPHA = 0.85
    BAR_COLOR = "0.55"
    BAR_W = step * 0.85

    AXIS_LABEL_SIZE = 13
    TICK_SIZE = 10
    LEGEND_SIZE = 9

    # localized span in percent coordinates
    s_x = float(x[int(safety_start)])
    e_x = float(x[int(safety_end)])
    span_l = max(0.0, s_x - step / 2.0)
    span_r = min(100.0, e_x + step / 2.0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(FIG_W, FIG_H),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.03},
    )

    # Slightly more breathing room on left/right to avoid label/tick clipping
    fig.subplots_adjust(left=0.145, right=0.985, top=0.985, bottom=0.18, hspace=0.03)

    # ---- Top panel ----
    ax1.plot(x, mean_cos_NN, linewidth=LINE_W, color="tab:blue", label="Safe-Safe")
    ax1.fill_between(x, mean_cos_NN - std_cos_NN, mean_cos_NN + std_cos_NN, alpha=FILL_ALPHA, color="tab:blue")

    ax1.plot(x, mean_cos_NM, linewidth=LINE_W, color="tab:orange", label="Safe-Unsafe")
    ax1.fill_between(x, mean_cos_NM - std_cos_NM, mean_cos_NM + std_cos_NM, alpha=FILL_ALPHA, color="tab:orange")

    ax1.axvspan(span_l, span_r, alpha=SPAN_ALPHA, label="localized range")
    ax1.axvline(s_x, linestyle="--", linewidth=1.2, color="red")
    ax1.axvline(e_x, linestyle="--", linewidth=1.2, color="red")

    ax1.set_ylabel("Cosine similarity", fontsize=AXIS_LABEL_SIZE)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # robust tight y-limits
    means_all = np.concatenate([mean_cos_NN.astype(np.float64), mean_cos_NM.astype(np.float64)], axis=0)
    lo = float(np.quantile(means_all, 0.02))
    hi = float(np.quantile(means_all, 0.98))
    smax = float(np.max(np.concatenate([std_cos_NN, std_cos_NM], axis=0))) if L > 0 else 0.0

    ylim_low = lo - 1.15 * smax
    ylim_high = hi + 1.15 * smax

    rng = max(1e-6, ylim_high - ylim_low)
    pad = 0.03 * rng
    ylim_low -= pad
    ylim_high += pad

    ax1.set_ylim(ylim_low, ylim_high)
    ax1.set_yticks(_inner_ticks(ylim_low, ylim_high, n=4))

    # legend location controlled by caller
    ax1.legend(loc=legend_loc, fontsize=LEGEND_SIZE, frameon=True)

    # ---- Bottom panel ----
    ax2.bar(x, gap, width=BAR_W, alpha=BAR_ALPHA, color=BAR_COLOR, edgecolor="none", align="center")

    ax2.axvspan(span_l, span_r, alpha=SPAN_ALPHA)
    ax2.axvline(s_x, linestyle="--", linewidth=1.2, color="red")
    ax2.axvline(e_x, linestyle="--", linewidth=1.2, color="red")

    ax2.set_xlabel("Layer", fontsize=AXIS_LABEL_SIZE)
    ax2.set_ylabel("Gap", fontsize=AXIS_LABEL_SIZE)
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    bot_low = float(np.min(gap))
    bot_high = float(np.max(gap))
    bot_rng = max(1e-6, bot_high - bot_low)
    bpad = 0.08 * bot_rng
    by0 = bot_low - bpad
    by1 = bot_high + bpad
    ax2.set_ylim(by0, by1)
    ax2.set_yticks(_inner_ticks(by0, by1, n=2))

    # avoid clipping of 0/100 tick labels: tiny adaptive x padding
    xpad = 0.55 * step
    ax2.set_xlim(0.0 - xpad, 100.0 + xpad)
    ax2.set_xticks([0, 25, 50, 75, 100])

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_mean_angle_diff_with_safety_range(
    mean_angle_diff: np.ndarray,
    safety_start: int,
    safety_end: int,
    save_path: str,
):
    L = len(mean_angle_diff)
    x = _layer_percent_axis(L)
    step = _percent_step(L)

    s_x = float(x[int(safety_start)])
    e_x = float(x[int(safety_end)])
    span_l = max(0.0, s_x - step / 2.0)
    span_r = min(100.0, e_x + step / 2.0)

    plt.figure(figsize=(4.8, 2.6))
    plt.plot(x, mean_angle_diff, linewidth=1.8)

    plt.axvspan(span_l, span_r, alpha=0.14, label="localized range")
    plt.axvline(s_x, linestyle="--", linewidth=1.2, color="red")
    plt.axvline(e_x, linestyle="--", linewidth=1.2, color="red")

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Mean Angle Diff", fontsize=12)

    ax = plt.gca()
    ax.grid(True, alpha=0.14)
    ax.tick_params(axis="both", which="major", labelsize=10)

    xpad = 0.55 * step
    ax.set_xlim(0.0 - xpad, 100.0 + xpad)
    ax.set_xticks([0, 25, 50, 75, 100])

    plt.legend(loc="best", fontsize=9, frameon=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ============================================================
# Main
# ============================================================

def main(
    data_path: str,
    model_path: str,
    save_dir: str,
    r: int = 20,
    max_length: int = 512,
    min_w: int = 5,                    
    rounds: int = 8,                    
    proposals_per_round: int = 10,      
    seed_base: int = 10,
    allow_same_within_class: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Reading data file (JSON): {data_path}")
    normal_cots, malicious_cots = load_cot_json(data_path)
    print(f"Normal sample count: {len(normal_cots)}")
    print(f"Malicious sample count: {len(malicious_cots)}")

    print(f"Loading model: {model_path}")
    model, backbone, tokenizer = load_text_only_model(model_path)

    total_layers = None
    try:
        if hasattr(backbone, "layers"):
            total_layers = len(backbone.layers)
        elif hasattr(backbone, "h"):
            total_layers = len(backbone.h)
    except Exception:
        total_layers = None

    is_qwen = _is_qwen_family(model_path)
    is_llama = _is_llama_family(model_path)

    # Legend rule:
    #   - Llama series: lower right
    #   - Qwen series:  upper left
    if is_llama:
        legend_loc = "lower right"
    elif is_qwen:
        legend_loc = "upper left"
    else:
        legend_loc = "best"

    if total_layers is not None:
        eff_layers = total_layers - 1 if (is_qwen and total_layers > 1) else total_layers
        if is_qwen and total_layers > 1:
            print(f"Model loaded, total {total_layers} layers (excluding embedding), Qwen effective layers after removing last = {eff_layers}")
        else:
            print(f"Model loaded, total {total_layers} layers (excluding embedding)")
    else:
        eff_layers = None
        print("Model loaded (unable to auto-detect layer count)")

    print("\nStarting cosine similarity computation...")

    r_eff = int(r)

    print(f"\n[1/3] Computing Normal-Normal pairs (r={r_eff})")
    allcos_NN = get_r_lists_cossim(
        backbone, tokenizer,
        normal_cots, normal_cots,
        seed=seed_base, r=r_eff,
        max_length=max_length,
        allow_same_within_class=allow_same_within_class
    )

    print(f"\n[2/3] Computing Malicious-Malicious pairs (r={r_eff})")
    allcos_MM = get_r_lists_cossim(
        backbone, tokenizer,
        malicious_cots, malicious_cots,
        seed=seed_base + 100, r=r_eff,
        max_length=max_length,
        allow_same_within_class=allow_same_within_class
    )

    print(f"\n[3/3] Computing Normal-Malicious pairs (r={r_eff})")
    allcos_NM = get_r_lists_cossim(
        backbone, tokenizer,
        normal_cots, malicious_cots,
        seed=seed_base + 1000, r=r_eff,
        max_length=max_length,
        allow_same_within_class=allow_same_within_class
    )

    # ===== Qwen: last layer does not participate in ANYTHING =====
    if is_qwen:
        print("\n[Qwen] Detected Qwen series: last layer excluded from statistics/localization/scaling/plotting (directly removed)")
        allcos_NN = _trim_last_layer(allcos_NN)
        allcos_MM = _trim_last_layer(allcos_MM)
        allcos_NM = _trim_last_layer(allcos_NM)

    print("\nCosine similarity computation completed!")

    results = {
        "Normal_Normal_pairs": allcos_NN,
        "Malicious_Malicious_pairs": allcos_MM,
        "Normal_Malicious_pairs": allcos_NM,
    }
    pickle_path = os.path.join(save_dir, "all_cos.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Cosine similarity data saved to: {pickle_path}")

    layer_analysis, mean_angle_diff, mean_cos_NN, mean_cos_NM = build_layer_analysis(allcos_NN, allcos_NM)

    # ========================================================
    # Localization MUST be based on GAP (largest consecutive 3)
    # ========================================================
    print("\nFixed selection: three consecutive layers with largest Gap (w=3)...")
    fixed_w = 3
    gap_signal = (np.array(mean_cos_NN, dtype=np.float32) - np.array(mean_cos_NM, dtype=np.float32))
    score, s, e = best_window_for_w(gap_signal, fixed_w)

    best = {
        "w": int(fixed_w),
        "score": float(score),
        "start": int(s),
        "end": int(e),
        "signal_len": int(len(gap_signal)),
        "seed": int(seed_base + 9999),
        "note": "fixed w=3; choose argmax window mean(GAP = mean_cos_NN - mean_cos_NM)",
    }

    safety_layers = list(range(int(best["start"]), int(best["end"]) + 1))

    print("\nKey layer localization completed!")
    print(f"best_w = {best['w']}, best_score = {best['score']:.6f}")
    print(f"Safety layers: {safety_layers}")
    print(f"Safety layer range: {best['start']} – {best['end']}")

    safety_layers_result = {
        "model_name": model_path,
        "total_layers_effective": int(eff_layers) if eff_layers is not None else -1,
        "qwen_drop_last_layer": bool(is_qwen),
        "safety_layers": safety_layers,
        "safety_layer_range": {"start": int(best["start"]), "end": int(best["end"])},
        "description": "Based on GAP=mean_cos_NN-mean_cos_NM signal, fixed w=3, select consecutive interval with largest mean GAP",
        "method": "signal=GAP(mean_cos_NN-mean_cos_NM); fixed w=3; choose argmax window mean(signal)",
        "parameters": {
            "r": r_eff,
            "max_length": max_length,
            "fixed_w": int(fixed_w),
            "seed_base": int(seed_base),
            "allow_same_within_class": bool(allow_same_within_class),
        },
        "best_search": best,
        "layer_analysis": layer_analysis,
    }

    json_path = os.path.join(save_dir, "safety_layers.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safety_layers_result, f, indent=2, ensure_ascii=False)
    print(f"Key layer localization results saved to: {json_path}")

    summary = {
        "model": Path(model_path).name,
        "total_layers_effective": int(eff_layers) if eff_layers is not None else -1,
        "qwen_drop_last_layer": bool(is_qwen),
        "safety_layers": safety_layers,
        "safety_layer_range": f"Layer {best['start']} - {best['end']}",
        "best_w": int(best["w"]),
        "best_score": float(best["score"]),
    }

    summary_path = os.path.join(save_dir, "safety_layers_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")

    # ===== Plot: mean_angle_diff (optional) =====
    vis_dir = os.path.join(save_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    plot_angle = os.path.join(vis_dir, "mean_angle_diff_with_localized_range.png")
    plot_mean_angle_diff_with_safety_range(
        mean_angle_diff=np.array(mean_angle_diff),
        safety_start=int(best["start"]),
        safety_end=int(best["end"]),
        save_path=plot_angle,
    )
    print(f"Plot saved: {plot_angle}")

    # ===== Two-panel plots: raw + scaled =====
    std_cos_NN = np.array(layer_analysis["std_cos_NN"], dtype=np.float32)
    std_cos_NM = np.array(layer_analysis["std_cos_NM"], dtype=np.float32)
    mean_cos_NN = np.array(mean_cos_NN, dtype=np.float32)
    mean_cos_NM = np.array(mean_cos_NM, dtype=np.float32)

    # (A) raw
    plot_raw = os.path.join(vis_dir, "cosine_two_panel_raw.png")
    plot_two_panel_cosine_and_gap(
        mean_cos_NN=mean_cos_NN,
        std_cos_NN=std_cos_NN,
        mean_cos_NM=mean_cos_NM,
        std_cos_NM=std_cos_NM,
        safety_start=int(best["start"]),
        safety_end=int(best["end"]),
        save_path=plot_raw,
        legend_loc=legend_loc,
    )
    print(f"Plot saved: {plot_raw}")

    # (B) scaled
    nn2, nn_std2, nm2, nm_std2, gap2, gamma = compute_global_math_scaled_curves(
        mean_cos_NN=mean_cos_NN,
        std_cos_NN=std_cos_NN,
        mean_cos_NM=mean_cos_NM,
        std_cos_NM=std_cos_NM,
        q=0.95,
    )

    plot_scaled = os.path.join(vis_dir, "cosine_two_panel_global_math_scaled.png")
    plot_two_panel_cosine_and_gap(
        mean_cos_NN=nn2,
        std_cos_NN=nn_std2,
        mean_cos_NM=nm2,
        std_cos_NM=nm_std2,
        safety_start=int(best["start"]),
        safety_end=int(best["end"]),
        save_path=plot_scaled,
        gap_override=gap2,
        legend_loc=legend_loc,
    )
    print(f"Plot saved: {plot_scaled}  (gamma={gamma:.4f})")

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print(f"Model: {Path(model_path).name}")
    if eff_layers not in (None, -1):
        print(f"Effective layers: {eff_layers}")
        print(f"Key layer percentage: {len(safety_layers) / max(1, int(eff_layers)) * 100:.1f}%")
    print(f"Key layer range: {best['start']} – {best['end']} (w={best['w']})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cosine sim + fixed window(w=3) localization by GAP; Qwen drops last layer everywhere; compact paper-friendly plots; legend rule: Llama lower-right, Qwen upper-left",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_path", type=str, required=True, help="Input JSON path")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--r", type=int, default=20, help="pairs per type (NN/MM/NM)")
    parser.add_argument("--max_length", type=int, default=512, help="token truncation length")

    parser.add_argument("--min_w", type=int, default=5, help="(unused) minimum window size w")
    parser.add_argument("--rounds", type=int, default=8, help="(unused) SA rounds")
    parser.add_argument("--proposals_per_round", type=int, default=10, help="(unused) SA proposals")

    parser.add_argument("--seed_base", type=int, default=10, help="random seed base")
    parser.add_argument(
        "--allow_same_within_class",
        action="store_true",
        help="If class has <2 samples, allow pairing same text with itself (avoid crash).",
    )

    args = parser.parse_args()

    # Resolve paths
    data_path = os.path.join(PROJECT_ROOT, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    model_path = os.path.join(PROJECT_ROOT, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    save_dir = os.path.join(PROJECT_ROOT, args.save_dir) if not os.path.isabs(args.save_dir) else args.save_dir

    main(
        data_path=data_path,
        model_path=model_path,
        save_dir=save_dir,
        r=args.r,
        max_length=args.max_length,
        min_w=args.min_w,
        rounds=args.rounds,
        proposals_per_round=args.proposals_per_round,
        seed_base=args.seed_base,
        allow_same_within_class=args.allow_same_within_class,
    )
