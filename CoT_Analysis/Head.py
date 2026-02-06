#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT Analysis Pipeline - Head-level Analysis

This script performs attention head-level analysis on processed CoT data.
"""

import os
import re
import json
import math
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

# Get script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ============================================================
# Hyperparameters 
# ============================================================
# --- GPU scheduling ---
GPU_PAIRS_DEFAULT = "0,1,2,3"   

# --- Data cap (your request: N=30, M=30) ---
MAX_N_DEFAULT = 30
MAX_M_DEFAULT = 30

# --- Logging ---
PRINT_EVERY_DEFAULT = 10   # print once per k samples processed

# --- Model / attention extraction ---
MAX_LENGTH_DEFAULT = 1024
ATTN_SIZE_DEFAULT  = 96
ROW_K_DEFAULT      = 24
PI_ITERS_DEFAULT   = 20

# --- Pair sampling ---
R_PAIRS_DEFAULT = 120
SEED_DEFAULT    = 10
ALLOW_SAME_WITHIN_DEFAULT = False

# --- Plot scaling / key head detection ---
Q_SCALE_DEFAULT   = 0.90
KEY_RATIO_DEFAULT = 0.70


PLOT_FIG_W = 4.8
PLOT_FIG_H = 2.6   # Wider horizontal, thinner vertical single plot


FONT_FAMILY = "serif"
FONT_SERIF_LIST = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]

LABEL_FONTSIZE  = 11
TICK_FONTSIZE   = 10
LEGEND_FONTSIZE = 10
TITLE_FONTSIZE  = 11

AXIS_LABEL_WEIGHT = "normal"
TICK_LABEL_WEIGHT = "normal"
LEGEND_WEIGHT     = "normal"

# Sparse ticks 
X_TICK_NBINS = 6
Y_TICK_NBINS = 5

# Line/fill/grid intensity (thin lines + light grid)
LINEWIDTH = 1.6
FILL_ALPHA = 0.12
GRID_ALPHA = 0.25


# ============================================================
# Config - Paths relative to project root
# ============================================================
MODEL_MAP = {
    "Llama_8B": "LLM/Llama-3-8B-Instruct",
    "Qwen_4B":  "LLM/Qwen3-4B-Thinking-2507",
    "Qwen_8B":  "LLM/Qwen3-VL-8B-Thinking",
}

DATA_WITH_SAFETY_DIR = "Data/Processed"  # Processed data directory
SAFETY_LAYER_DIR     = "Data/Safety_Layer"  # Safety layer analysis results
OUT_ROOT             = "Data/Operator_Analysis"  # Output directory for head analysis

DEFAULT_JSONS = [
    f"{DATA_WITH_SAFETY_DIR}/Llama_8B_d_ori.json",
    f"{DATA_WITH_SAFETY_DIR}/Llama_8B_i_ori.json",
    f"{DATA_WITH_SAFETY_DIR}/Qwen_4B_d_ori.json",
    f"{DATA_WITH_SAFETY_DIR}/Qwen_4B_i_ori.json",
    f"{DATA_WITH_SAFETY_DIR}/Qwen_8B_d_ori.json",
    f"{DATA_WITH_SAFETY_DIR}/Qwen_8B_i_ori.json",
]

# GPU Configuration
GPU_IDS = ""  # e.g., "0,1" or "" for auto

# y-axis names
B1_NAME = "B1"
# B2 needs to scalarize direction vector for display
B2_NAME = "B2"
B3_NAME = "B3"

# ===============================
# Qwen3-VL support (modelscope)
# ===============================
try:
    from modelscope import Qwen3VLForConditionalGeneration
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False


# ============================================================
# Helpers: JSON loading
# ============================================================
def _get_nested(d: Dict[str, Any], path: str):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_cot_items(
    json_path: str,
    cot_field: str = "CoT",
    toxicity_field: str = "is_cot_toxicity",
    toxic_value: int = 1,
    id_field: str = "id",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Return:
      normals   : list[(id, cot)]
      malicious : list[(id, cot)]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normals, malicious = [], []
    for item in data:
        cot = item.get(cot_field, "")
        lab = _get_nested(item, toxicity_field)
        sid = item.get(id_field, None)
        if sid is None:
            sid = str(item.get("index", ""))

        sid = str(sid)
        if not isinstance(cot, str) or not cot.strip():
            continue
        if lab is None:
            continue

        if int(lab) == int(toxic_value):
            malicious.append((sid, cot.strip()))
        else:
            normals.append((sid, cot.strip()))

    return normals, malicious


def infer_model_key_from_json(json_path: str) -> str:
    base = Path(json_path).name
    m = re.match(r"^(Llama_8B|Qwen_4B|Qwen_8B)_", base)
    if not m:
        raise ValueError(f"Cannot infer model key from filename: {base}")
    return m.group(1)


def safety_layers_summary_path(json_path: str) -> str:
    base = Path(json_path).stem
    safety_dir = os.path.join(PROJECT_ROOT, SAFETY_LAYER_DIR) if not os.path.isabs(SAFETY_LAYER_DIR) else SAFETY_LAYER_DIR
    return os.path.join(safety_dir, base, "safety_layers_summary.json")


def load_safety_layers(summary_json_path: str) -> List[int]:
    with open(summary_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    layers = obj.get("safety_layers", [])
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError(f"Invalid safety_layers in {summary_json_path}")
    return [int(x) for x in layers]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ============================================================
# Model loading (force eager attention!)
# ============================================================
def _looks_like_qwen3_vl(model_path: str) -> bool:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        mt = getattr(cfg, "model_type", "")
        cls = cfg.__class__.__name__.lower()
        return ("qwen3_vl" in str(mt).lower()) or ("qwen3vl" in cls) or ("qwen3_vl" in cls)
    except Exception:
        return "qwen3-vl" in model_path.lower() or "qwen3_vl" in model_path.lower()


def _force_eager_in_config(cfg):
    # Different transformer versions use different fields: try all
    for k in ["attn_implementation", "_attn_implementation"]:
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, "eager")
            except Exception:
                pass
    return cfg


def load_text_only_backbone(model_path: str):
    """
    Return (model_wrapper, backbone, tokenizer)
    backbone: transformer body, supports output_attentions
    """
    import torch
    from transformers import AutoTokenizer, AutoConfig

    # These three lines reduce SDPA/Flash attention interference (double insurance)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = _force_eager_in_config(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_vl = _looks_like_qwen3_vl(model_path)

    if is_vl:
        if not HAS_QWEN3VL:
            raise RuntimeError(
                "Detected Qwen3-VL model but modelscope not installed.\n"
                "Please `pip install modelscope`."
            )
        print("[INFO] Loading Qwen3-VL via modelscope (force eager if possible)")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        # Try to write eager back to config
        try:
            _force_eager_in_config(model.config)
        except Exception:
            pass
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError("Qwen3VLForConditionalGeneration has no `.model` attribute.")
    else:
        from transformers import AutoModelForCausalLM
        print("[INFO] Loading standard CausalLM via transformers (attn_implementation=eager)")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        # Write again (double insurance)
        try:
            _force_eager_in_config(model.config)
        except Exception:
            pass

        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError("AutoModelForCausalLM has no `.model` attribute for this checkpoint.")

    model.eval()
    backbone.eval()
    return model, backbone, tokenizer


# ============================================================
# Attention utilities
# ============================================================
def get_attentions_for_text(backbone, tokenizer, text: str, max_length: int):
    import torch

    with torch.no_grad():
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
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

        atts = getattr(outputs, "attentions", None)
        if atts is None:
            raise RuntimeError(
                "outputs.attentions is None. You are likely still using SDPA/Flash attention.\n"
                "Fix: force eager attention (attn_implementation='eager') when loading the model."
            )
        return atts


def downsample_attn(attn_2d, target: int):
    import torch
    import torch.nn.functional as F

    x = attn_2d.float()
    x4 = x.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    y4 = F.interpolate(x4, size=(target, target), mode="bilinear", align_corners=False)
    y = y4[0, 0].contiguous()
    y = torch.clamp(y, min=0.0)
    y = y / (y.sum(dim=-1, keepdim=True) + 1e-12)
    return y.cpu()


# ============================================================
# Operator metrics (softmax Jacobian row form)
# ============================================================
def j_matvec(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    pv = float(np.dot(p, v))
    return p * v - p * pv


def power_iteration_j(p: np.ndarray, iters: int = 20, seed: int = 0) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = p.size
    v = rng.normal(size=n).astype(np.float64)
    v = v - v.mean()
    v /= (np.linalg.norm(v) + 1e-12)

    for _ in range(iters):
        w = j_matvec(p, v)
        nw = np.linalg.norm(w)
        if nw < 1e-14:
            break
        v = w / (nw + 1e-12)

    w = j_matvec(p, v)
    lam = float(np.dot(v, w))
    return lam, v.astype(np.float32)


def effective_rank_from_eigs(eigs: np.ndarray, eps: float = 1e-12) -> float:
    e = np.asarray(eigs, dtype=np.float64)
    e = np.clip(e, 0.0, None)
    w = (e * e)
    s = float(np.sum(w))
    if s <= eps:
        return 0.0
    w = w / s
    w = np.clip(w, eps, 1.0)
    return float(np.exp(-np.sum(w * np.log(w))))


def operator_features_from_attn(
    A: np.ndarray,
    row_k: int = 24,
    pi_iters: int = 20,
    seed: int = 0,
) -> Tuple[float, np.ndarray, float]:
    n = A.shape[0]
    idx = np.linspace(0, n - 1, num=min(row_k, n), dtype=int)

    best_b1 = -1e18
    best_v = None
    best_b3 = 0.0

    for t in idx:
        p = A[t].astype(np.float64)
        p = np.clip(p, 0.0, None)
        p = p / (p.sum() + 1e-12)

        lam, v = power_iteration_j(p, iters=pi_iters, seed=seed + int(t))

        J = np.diag(p) - np.outer(p, p)
        eigs = np.linalg.eigvalsh(J).astype(np.float64)
        b3 = effective_rank_from_eigs(eigs)

        if lam > best_b1:
            best_b1 = float(lam)
            v = v.astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            best_v = v
            best_b3 = float(b3)

    if best_v is None:
        best_v = np.zeros((n,), dtype=np.float32)

    return float(best_b1), best_v, float(best_b3)


# ============================================================
# Scaling (your strategy)
# ============================================================
def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


def global_adaptive_power_scale_signed(x: np.ndarray, q: float = 0.90, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    t = np.abs(x)

    m = float(np.max(t)) if t.size > 0 else 0.0
    if not np.isfinite(m) or m < 1e-30:
        return x.copy(), 1.0

    p = _safe_quantile(t, q)
    r = float(p / (m + eps))
    if (not np.isfinite(r)) or (r <= 0.0) or (r >= 1.0 - 1e-6):
        gamma = 1.0
    else:
        gamma = float(np.log(0.5) / np.log(r))
        if (not np.isfinite(gamma)) or (gamma < 1.0):
            gamma = 1.0

    u = t / (m + eps)
    y = np.sign(x) * (u ** gamma) * m
    return y.astype(np.float32), float(gamma)


def plot_nn_nm_scaled_clean(
    mean_nn: np.ndarray,
    std_nn: np.ndarray,
    mean_nm: np.ndarray,
    std_nm: np.ndarray,
    save_path: str,
    ylabel: str,
    q: float = 0.90,
    key_ratio: float = 0.95,
    pad_ratio: float = 0.08,
):
    """
    Plot with legend at best location (saved PNG will show it).
    
    Modifications:
      - Legend N/M renamed to Safe/Unsafe
      - Sparse x/y ticks
      - Larger, bold font, Times New Roman
      - X/Y axis labels also bold, Times New Roman
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # global rcParams 
    # ---- style: match simple paper figure ----
    plt.rcParams.update({
        "font.family": FONT_FAMILY,              # "serif"
        "font.serif": FONT_SERIF_LIST,           # fallback chain
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelweight": AXIS_LABEL_WEIGHT,   # "normal"
        "axes.titleweight": AXIS_LABEL_WEIGHT,   # "normal"
        "font.weight": "normal",
    })


    mean_nn = np.asarray(mean_nn, dtype=np.float32)
    mean_nm = np.asarray(mean_nm, dtype=np.float32)
    std_nn  = np.asarray(std_nn,  dtype=np.float32)
    std_nm  = np.asarray(std_nm,  dtype=np.float32)
    assert mean_nn.shape == mean_nm.shape == std_nn.shape == std_nm.shape

    H = mean_nn.size
    x = np.arange(H)

    mid = (mean_nn + mean_nm) / 2.0
    gap = (mean_nn - mean_nm)

    gap2, gamma = global_adaptive_power_scale_signed(gap, q=q)
    nn2 = mid + gap2 / 2.0
    nm2 = mid - gap2 / 2.0

    gmax = float(np.max(np.abs(gap))) + 1e-12
    u = (np.abs(gap) / gmax).astype(np.float32)
    std_scale = (u ** max(0.0, (gamma - 1.0))).astype(np.float32)
    nn_std2 = std_nn * std_scale
    nm_std2 = std_nm * std_scale

    a = np.abs(gap2)
    amax = float(np.max(a)) if a.size else 0.0
    if amax > 0:
        thr = key_ratio * amax
        key_heads = np.where(a >= thr)[0].tolist()
    else:
        thr = 0.0
        key_heads = []

    y_low = float(np.min([np.min(nn2 - nn_std2), np.min(nm2 - nm_std2)]))
    y_high = float(np.max([np.max(nn2 + nn_std2), np.max(nm2 + nm_std2)]))
    if (not np.isfinite(y_low)) or (not np.isfinite(y_high)) or (y_high <= y_low + 1e-12):
        y_low, y_high = 0.0, 1.0

    y_rng = y_high - y_low
    y_low -= pad_ratio * y_rng
    y_high += pad_ratio * y_rng

    fig = plt.figure(figsize=(PLOT_FIG_W, PLOT_FIG_H))
    ax = plt.gca()

    # labels: N->Safe, M->Unsafe
    ax.plot(x, nn2, linewidth=LINEWIDTH, label="Safe")
    ax.fill_between(x, nn2 - nn_std2, nn2 + nn_std2, alpha=FILL_ALPHA)

    ax.plot(x, nm2, linewidth=LINEWIDTH, label="Unsafe")
    ax.fill_between(x, nm2 - nm_std2, nm2 + nm_std2, alpha=FILL_ALPHA)

    for h in key_heads:
        ax.axvline(h, linestyle="--", linewidth=2.0, color="red", alpha=0.9)

    ax.set_xlabel("Head", fontsize=LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT, fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight=AXIS_LABEL_WEIGHT, fontfamily=FONT_FAMILY)

    ax.set_xlim(-0.5, H - 0.5)
    ax.set_ylim(y_low, y_high)

    # sparse ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_NBINS, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_TICK_NBINS))

    # tick font
    for tick in ax.get_xticklabels():
        tick.set_fontsize(TICK_FONTSIZE)
        tick.set_fontweight(TICK_LABEL_WEIGHT)
        tick.set_fontfamily(FONT_FAMILY)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(TICK_FONTSIZE)
        tick.set_fontweight(TICK_LABEL_WEIGHT)
        tick.set_fontfamily(FONT_FAMILY)

    leg = ax.legend(frameon=False, loc="best", fontsize=LEGEND_FONTSIZE)
    for txt in leg.get_texts():
        txt.set_fontweight(LEGEND_WEIGHT)
        txt.set_fontfamily(FONT_FAMILY)

    ax.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return {"gamma": gamma, "key_heads": key_heads, "thr": float(thr)}


# ============================================================
# Pair sampling 
# ============================================================
def sample_pairs_same_pool(items: List[int], r: int, rng: random.Random, allow_same: bool) -> List[Tuple[int, int]]:
    pairs = []
    n = len(items)
    if n == 0:
        return pairs
    for _ in range(r):
        if n >= 2:
            a, b = rng.sample(items, 2)
        else:
            if not allow_same:
                break
            a = items[0]; b = items[0]
        pairs.append((a, b))
    return pairs


def sample_pairs_cross_pool(items_a: List[int], items_b: List[int], r: int, rng: random.Random) -> List[Tuple[int, int]]:
    pairs = []
    if len(items_a) == 0 or len(items_b) == 0:
        return pairs
    for _ in range(r):
        a = rng.choice(items_a)
        b = rng.choice(items_b)
        pairs.append((a, b))
    return pairs


# ============================================================
# Core analysis for one json
# ============================================================
def analyze_one_json(
    json_path: str,
    out_root: str,
    max_length: int,
    attn_size: int,
    row_k: int,
    pi_iters: int,
    r_pairs: int,
    seed: int,
    max_n: int,
    max_m: int,
    allow_same_within: bool,
    q_scale: float,
    key_ratio: float,
    print_every: int,
):
    import torch  # ensure in child after CUDA_VISIBLE_DEVICES

    json_base = Path(json_path).stem
    # Resolve output root
    if not os.path.isabs(out_root):
        out_root = os.path.join(PROJECT_ROOT, out_root)
    out_dir = os.path.join(out_root, json_base)
    ensure_dir(out_dir)

    sl_path = safety_layers_summary_path(json_path)
    safety_layers = load_safety_layers(sl_path)
    print(f"[INFO] {json_base}: safety_layers = {safety_layers}")

    model_key = infer_model_key_from_json(json_path)
    model_path = MODEL_MAP[model_key]
    # Resolve model path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    print(f"[INFO] {json_base}: model = {model_key} ({model_path})")

    _, backbone, tokenizer = load_text_only_backbone(model_path)

    normals, malicious = load_cot_items(json_path)
    rng = random.Random(seed)

    # your request: N=30, M=30 (cap separately)
    if len(normals) > max_n:
        normals = rng.sample(normals, max_n)
    else:
        normals = normals[:max_n]

    if len(malicious) > max_m:
        malicious = rng.sample(malicious, max_m)
    else:
        malicious = malicious[:max_m]

    print(f"[INFO] {json_base}: N={len(normals)} M={len(malicious)} (capN={max_n}, capM={max_m})")

    all_samples = normals + malicious
    if len(all_samples) == 0:
        raise RuntimeError(f"{json_base}: empty samples after filtering/capping.")

    labels = [0] * len(normals) + [1] * len(malicious)
    texts = [txt for _, txt in all_samples]

    idx_N = [i for i, y in enumerate(labels) if y == 0]
    idx_M = [i for i, y in enumerate(labels) if y == 1]
    if len(idx_N) == 0 or len(idx_M) == 0:
        raise RuntimeError(f"{json_base}: empty group (N={len(idx_N)} M={len(idx_M)})")

    # probe to infer shapes
    probe_atts = get_attentions_for_text(backbone, tokenizer, texts[0], max_length=max_length)
    num_layers = len(probe_atts)
    for L in safety_layers:
        if L < 0 or L >= num_layers:
            raise ValueError(f"{json_base}: safety layer {L} out of range (0..{num_layers-1})")
    num_heads = int(probe_atts[safety_layers[0]].shape[1])

    Ls = len(safety_layers)
    S = len(all_samples)

    B1 = np.zeros((S, Ls, num_heads), dtype=np.float32)
    B3 = np.zeros((S, Ls, num_heads), dtype=np.float32)
    B2 = np.zeros((S, Ls, num_heads, attn_size), dtype=np.float16)

    print(f"[INFO] {json_base}: extracting features... (S={S}, layers={Ls}, heads={num_heads})")

    for i, txt in enumerate(texts):
        done = i + 1
        if done == 1 or (done % max(1, print_every) == 0) or (done == S):
            print(f"[{json_base}] processed {done}/{S}")

        atts = get_attentions_for_text(backbone, tokenizer, txt, max_length=max_length)

        for li, L in enumerate(safety_layers):
            attL = atts[L][0]  # (heads, T, T)
            for h in range(num_heads):
                A = attL[h]
                A_ds = downsample_attn(A, target=attn_size)
                A_np = A_ds.numpy()

                b1, v, b3 = operator_features_from_attn(
                    A=A_np,
                    row_k=row_k,
                    pi_iters=pi_iters,
                    seed=seed + i * 17 + li * 1000 + h * 13,
                )
                B1[i, li, h] = b1
                B3[i, li, h] = b3

                if v.shape[0] != attn_size:
                    v = np.resize(v, (attn_size,)).astype(np.float32)
                    v = v / (np.linalg.norm(v) + 1e-12)
                B2[i, li, h, :] = v.astype(np.float16)

    run_summary = {
        "json": json_path,
        "model_key": model_key,
        "model_path": model_path,
        "safety_layers": safety_layers,
        "max_length": max_length,
        "attn_size": attn_size,
        "row_k": row_k,
        "pi_iters": pi_iters,
        "r_pairs": r_pairs,
        "allow_same_within": allow_same_within,
        "seed": seed,
        "q_scale": q_scale,
        "key_ratio": key_ratio,
        "max_n": max_n,
        "max_m": max_m,
        "mode": "raw_group_values(N/M)",
        "B2_scalar": "centroid(sum k*|v_k| / sum |v_k|)",
        "legend": {"N": "Safe", "M": "Unsafe"},
        "tick_sparse": {"x_nbins": X_TICK_NBINS, "y_nbins": Y_TICK_NBINS},
        "font": {"family": FONT_FAMILY, "label_size": LABEL_FONTSIZE, "tick_size": TICK_FONTSIZE},
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    def _dir_centroid(v: np.ndarray, eps: float = 1e-12) -> float:
        """
        Scalarize direction vector v (attn_size,) to "centroid position":
          c = sum(k * |v_k|) / sum(|v_k|)
        Returns range approximately [0, attn_size-1]
        """
        w = np.abs(v.astype(np.float32))
        s = float(np.sum(w))
        if s <= eps:
            return 0.0
        idx = np.arange(w.size, dtype=np.float32)
        return float(np.sum(idx * w) / (s + eps))

    def compute_group_stats(metric: str, li: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Directly display raw N/M values: compute group mean/std for each head.
        Returns: mean_N, std_N, mean_M, std_M  (shape = [num_heads])
        """
        if metric == "B1":
            XN = B1[idx_N, li, :]  # (N, H)
            XM = B1[idx_M, li, :]  # (M, H)
            return XN.mean(0), XN.std(0), XM.mean(0), XM.std(0)

        elif metric == "B3":
            XN = B3[idx_N, li, :]
            XM = B3[idx_M, li, :]
            return XN.mean(0), XN.std(0), XM.mean(0), XM.std(0)

        elif metric == "B2":
            # B2 stores direction vector (S, L, H, attn_size)
            # Need to scalarize before plotting "raw value" curves
            XN = np.zeros((len(idx_N), num_heads), dtype=np.float32)
            XM = np.zeros((len(idx_M), num_heads), dtype=np.float32)

            for ii, sidx in enumerate(idx_N):
                for h in range(num_heads):
                    XN[ii, h] = _dir_centroid(B2[sidx, li, h, :])

            for ii, sidx in enumerate(idx_M):
                for h in range(num_heads):
                    XM[ii, h] = _dir_centroid(B2[sidx, li, h, :])

            return XN.mean(0), XN.std(0), XM.mean(0), XM.std(0)

        else:
            raise ValueError(metric)

    # ============================================================
    # Collect key heads across layers (for later disturbance usage)
    # ============================================================
    key_heads_by_metric: Dict[str, Dict[int, List[int]]] = {"B1": {}, "B2": {}, "B3": {}}

    for li, L in enumerate(safety_layers):
        layer_dir = os.path.join(out_dir, "visualization", f"layer_{L:02d}")
        ensure_dir(layer_dir)

        # ---------------- B1 ----------------
        mean_N, std_N, mean_M, std_M = compute_group_stats("B1", li)
        meta = plot_nn_nm_scaled_clean(
            mean_N, std_N, mean_M, std_M,
            save_path=os.path.join(layer_dir, "B1.png"),
            ylabel=B1_NAME,
            q=q_scale,
            key_ratio=key_ratio,
        )
        np.savetxt(
            os.path.join(layer_dir, "B1.csv"),
            np.column_stack([mean_N, std_N, mean_M, std_M]),
            delimiter=",",
            header="mean_N,std_N,mean_M,std_M",
            comments=""
        )
        with open(os.path.join(layer_dir, "B1_key_heads.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        key_heads_by_metric["B1"][int(L)] = [int(h) for h in meta.get("key_heads", [])]

        # ---------------- B2 ----------------
        mean_N, std_N, mean_M, std_M = compute_group_stats("B2", li)
        meta = plot_nn_nm_scaled_clean(
            mean_N, std_N, mean_M, std_M,
            save_path=os.path.join(layer_dir, "B2.png"),
            ylabel=B2_NAME,
            q=q_scale,
            key_ratio=key_ratio,
        )
        np.savetxt(
            os.path.join(layer_dir, "B2.csv"),
            np.column_stack([mean_N, std_N, mean_M, std_M]),
            delimiter=",",
            header="mean_N,std_N,mean_M,std_M",
            comments=""
        )
        with open(os.path.join(layer_dir, "B2_key_heads.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        key_heads_by_metric["B2"][int(L)] = [int(h) for h in meta.get("key_heads", [])]

        # ---------------- B3 ----------------
        mean_N, std_N, mean_M, std_M = compute_group_stats("B3", li)
        meta = plot_nn_nm_scaled_clean(
            mean_N, std_N, mean_M, std_M,
            save_path=os.path.join(layer_dir, "B3.png"),
            ylabel=B3_NAME,
            q=q_scale,
            key_ratio=key_ratio,
        )
        np.savetxt(
            os.path.join(layer_dir, "B3.csv"),
            np.column_stack([mean_N, std_N, mean_M, std_M]),
            delimiter=",",
            header="mean_N,std_N,mean_M,std_M",
            comments=""
        )
        with open(os.path.join(layer_dir, "B3_key_heads.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        key_heads_by_metric["B3"][int(L)] = [int(h) for h in meta.get("key_heads", [])]

    # ============================================================
    # Write consolidated key_layers / key_heads summary
    # ============================================================
    key_layers = [int(x) for x in safety_layers]

    # union over metrics per layer
    key_heads_union: Dict[int, List[int]] = {}
    for L in key_layers:
        hs: List[int] = []
        for met in ["B1", "B2", "B3"]:
            hs += list(key_heads_by_metric.get(met, {}).get(int(L), []))
        key_heads_union[int(L)] = sorted(set(int(h) for h in hs))

    key_summary = {
        "json": json_path,
        "model_key": model_key,
        "model_path": model_path,
        "key_layers": key_layers,
        "key_heads_union": key_heads_union,              # each layer union heads
        "key_heads_by_metric": key_heads_by_metric,      # keep per-metric heads
        "key_ratio": float(key_ratio),
        "q_scale": float(q_scale),
        "plot_style": {
            "legend": {"N": "Safe", "M": "Unsafe"},
            "font_family": FONT_FAMILY,
            "label_fontsize": LABEL_FONTSIZE,
            "tick_fontsize": TICK_FONTSIZE,
            "x_tick_nbins": X_TICK_NBINS,
            "y_tick_nbins": Y_TICK_NBINS,
        },
    }

    # (1) save in Operator_Analysis output dir
    with open(os.path.join(out_dir, "key_heads_summary.json"), "w", encoding="utf-8") as f:
        json.dump(key_summary, f, indent=2, ensure_ascii=False)

    # (2) append into safety_layers_summary.json (historical file)
    try:
        with open(sl_path, "r", encoding="utf-8") as f:
            sl_obj = json.load(f)
        sl_obj["key_layers"] = key_layers
        sl_obj["key_heads_union"] = key_heads_union
        sl_obj["key_heads_by_metric"] = key_heads_by_metric
        sl_obj["key_ratio"] = float(key_ratio)
        sl_obj["q_scale"] = float(q_scale)
        sl_obj["legend"] = {"N": "Safe", "M": "Unsafe"}
        with open(sl_path, "w", encoding="utf-8") as f:
            json.dump(sl_obj, f, indent=2, ensure_ascii=False)
        print(f"[INFO] updated safety summary -> {sl_path}")
    except Exception as e:
        print(f"[WARN] failed to update safety summary {sl_path}: {e}")

    print(f"[DONE] {json_base} -> {out_dir}")


# ============================================================
# Parallel scheduler by GPU pairs 
# ============================================================
def parse_gpu_pairs(s: str) -> List[str]:
    """
    "0,1;2,3;4,5" -> ["0,1", "2,3", "4,5"]
    allow "0" as single gpu.
    """
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return parts


def _worker_entry(json_path: str, gpu_pair: str, kwargs: Dict[str, Any]):
    # bind GPUs for THIS process only 
    if gpu_pair is not None and str(gpu_pair).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_pair).strip()

    # optional: reduce tokenizer thread noise
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    analyze_one_json(json_path=json_path, **kwargs)


def run_jobs(jsons: List[str], gpu_pairs: List[str], kwargs: Dict[str, Any]):
    """
    If gpu_pairs empty -> sequential in current process.
    Else -> spawn processes, one json per process, each process binds to one gpu_pair.
    """
    if not gpu_pairs:
        for jp in jsons:
            _worker_entry(jp, gpu_pair=None, kwargs=kwargs)
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    pending = list(jsons)
    running: List[Tuple[mp.Process, str, str]] = []  # (proc, json, gpu_pair)

    def launch_one(jp: str, gp: str):
        p = ctx.Process(target=_worker_entry, args=(jp, gp, kwargs))
        p.start()
        return p

    # initial fill
    for gp in gpu_pairs:
        if not pending:
            break
        jp = pending.pop(0)
        p = launch_one(jp, gp)
        running.append((p, jp, gp))
        print(f"[SCHED] start {Path(jp).name} on GPUs [{gp}]")

    # loop
    while running:
        still_running = []
        for p, jp, gp in running:
            p.join(timeout=0.5)
            if p.is_alive():
                still_running.append((p, jp, gp))
            else:
                code = p.exitcode
                if code != 0:
                    raise RuntimeError(f"[SCHED] job failed: {Path(jp).name} on GPUs [{gp}] exitcode={code}")
                print(f"[SCHED] done  {Path(jp).name} on GPUs [{gp}]")

                if pending:
                    jp2 = pending.pop(0)
                    p2 = launch_one(jp2, gp)
                    still_running.append((p2, jp2, gp))
                    print(f"[SCHED] start {Path(jp2).name} on GPUs [{gp}]")
        running = still_running


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser("operator_analysis.py (raw N/M values + same scaling + gpu-pair parallel)")

    parser.add_argument("--jsons", type=str, default=",".join(DEFAULT_JSONS),
                        help="comma-separated json paths")
    parser.add_argument("--out_root", type=str, default=OUT_ROOT)

    parser.add_argument("--max_length", type=int, default=MAX_LENGTH_DEFAULT)
    parser.add_argument("--attn_size", type=int, default=ATTN_SIZE_DEFAULT)
    parser.add_argument("--row_k", type=int, default=ROW_K_DEFAULT)
    parser.add_argument("--pi_iters", type=int, default=PI_ITERS_DEFAULT)

    # kept for compatibility; raw-value mode does not use pair sampling
    parser.add_argument("--r_pairs", type=int, default=R_PAIRS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)

    # N=30, M=30 (separate)
    parser.add_argument("--max_n", type=int, default=MAX_N_DEFAULT)
    parser.add_argument("--max_m", type=int, default=MAX_M_DEFAULT)

    parser.add_argument("--allow_same_within", action="store_true", default=ALLOW_SAME_WITHIN_DEFAULT)

    parser.add_argument("--q_scale", type=float, default=Q_SCALE_DEFAULT)
    parser.add_argument("--key_ratio", type=float, default=KEY_RATIO_DEFAULT)
    parser.add_argument("--print_every", type=int, default=PRINT_EVERY_DEFAULT)

    # parallel by GPU pairs
    parser.add_argument("--gpu_pairs", type=str, default=GPU_PAIRS_DEFAULT,
                        help='e.g. "0,1;2,3;4,5;6,7"  (one process per pair). Empty => sequential')

    args = parser.parse_args()

    # Resolve output root
    if not os.path.isabs(args.out_root):
        args.out_root = os.path.join(PROJECT_ROOT, args.out_root)
    os.makedirs(args.out_root, exist_ok=True)

    # Resolve JSON paths
    jsons = []
    for json_path in [x.strip() for x in args.jsons.split(",") if x.strip()]:
        if not os.path.isabs(json_path):
            json_path = os.path.join(PROJECT_ROOT, json_path)
        jsons.append(json_path)
    
    gpu_pairs = parse_gpu_pairs(args.gpu_pairs)

    kwargs = dict(
        out_root=args.out_root,
        max_length=args.max_length,
        attn_size=args.attn_size,
        row_k=args.row_k,
        pi_iters=args.pi_iters,
        r_pairs=args.r_pairs,  
        seed=args.seed,
        max_n=args.max_n,
        max_m=args.max_m,
        allow_same_within=args.allow_same_within,  
        q_scale=args.q_scale,
        key_ratio=args.key_ratio,
        print_every=args.print_every,
    )

    run_jobs(jsons=jsons, gpu_pairs=gpu_pairs, kwargs=kwargs)


if __name__ == "__main__":
    main()
