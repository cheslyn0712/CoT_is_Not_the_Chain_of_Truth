<div align="center">

# CoT is Not the Chain of Truth

**An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation**

<p>
  <a href="https://icml.cc/virtual/2026/poster/61042" target="_blank" rel="noopener noreferrer">
    <img src="assets/icml-navbar-logo.svg" alt="ICML" height="28" align="middle"><strong> ICML 2026</strong>
  </a>
</p>

**Paper:** [ICML 2026 · PMLR](https://proceedings.mlr.press/v306/tong26e.html)

[![arXiv](https://img.shields.io/badge/arXiv-2602.04856-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.04856)
[![Project Page](https://img.shields.io/badge/Project-Page-102033?style=for-the-badge&logo=googlechrome&logoColor=white)](https://chunlingong.me/projects/cot-chain-of-truth/)
[![PDF](https://img.shields.io/badge/Paper-PDF-243447?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://raw.githubusercontent.com/mlresearch/v306/main/assets/tong26e/tong26e.pdf)

<p>
  <strong>Zhao Tong</strong><sup>*</sup> ·
  <strong>Chunlin Gong</strong><sup>*</sup> ·
  <strong>Yiping Zhang</strong> ·
  <strong>Haichao Shi</strong> ·
  <strong>Qiang Liu</strong> ·
  <strong>Xingcheng Xu</strong><sup>†</sup> ·
  <strong>Shu Wu</strong> ·
  <strong>Xiao-Yu Zhang</strong><sup>†</sup>
</p>
<p><sup>*</sup>Equal contribution: Zhao Tong, Chunlin Gong · <sup>†</sup>Corresponding author: Xingcheng Xu, Xiao-Yu Zhang</p>
<p>Institute of Information Engineering, CAS · University of Chinese Academy of Sciences · University of Minnesota · Shanghai AI Laboratory · Institute of Automation, CAS</p>

</div>

---

## Overview

We study chain-of-thought (CoT) safety in reasoning LLMs, connect unsafe reasoning traces and final-answer refusal to safety-related layers and attention heads, and evaluate targeted interventions on fake-news generation and HarmBench.

Chain-of-thought safety asks how harmful content develops during a reasoning model's generation process. CoT is Not the Chain of Truth examines cases where the generated reasoning trace contains unsafe content and the final answer refuses the request. Trace annotations and final-response labels make this relationship observable.

The paper connects this process-level safety question to mechanistic interpretability through layer localization, attention-head analysis, and Jacobian spectral measurements. Perturbation experiments examine the behavioral role of the identified components, and targeted parameter updates are evaluated on fake-news generation and HarmBench.

Research note: [Chain-of-Thought Safety and Mechanistic Interpretability](https://chunlingong.me/research/reasoning-trace-safety/)

## Code and resources

| Component | Entry point |
|---|---|
| Trace generation | [CoT_Generation/Generation.py](CoT_Generation/Generation.py) |
| Trace annotation | [CoT_Annotation/label.py](CoT_Annotation/label.py) |
| Annotation review | [CoT_Annotation/verify.py](CoT_Annotation/verify.py) |
| Layer analysis | [CoT_Analysis/Layer.py](CoT_Analysis/Layer.py) |
| Attention-head analysis | [CoT_Analysis/Head.py](CoT_Analysis/Head.py) |
| Data format | [Data/README.md](Data/README.md) |
| Model setup | [LLM/README.md](LLM/README.md) |

## Pipeline

Generation, trace labeling, and layer-to-head analysis are in `CoT_Generation/`, `CoT_Annotation/`, and `CoT_Analysis/`. The analysis localizes unsafe routing and supports the critical-head updates reported in the paper.

Three stages: **CoT_Generation** → **CoT_Annotation** → **CoT_Analysis**

```
Seed (Real_News.json)
  → Raw ({model}/{prompt_type}/{style}/news.json)
    → Processed ({model}/{prompt_type}/{style}/news.json)
      → HumanCheck ({model}/{prompt_type}/{style}/news.json)
```

### Quick Start

```bash
pip install -r requirements.txt

# 1. Generation
cd CoT_Generation && python Generation.py

# 2. Annotation
cd CoT_Annotation && python label.py && python verify.py

# 3. Analysis
cd CoT_Analysis && python Layer.py --data_path ... --model_path ... --save_dir ...
cd CoT_Analysis && python Head.py --jsons ... --out_root ...
```

## Directory Structure

```
CoT_is_Not_the_Chain_of_Truth/
├── CoT_Generation/     # Stage 1: Generate CoT data
├── CoT_Annotation/     # Stage 2: Label and verify
├── CoT_Analysis/       # Stage 3: Layer & head analysis
├── Data/               # Seed → Raw → Processed → HumanCheck
└── LLM/                # Language models (download separately)
```

See [Data/README.md](Data/README.md) and [LLM/README.md](LLM/README.md) for data format and model download instructions.

## Citation

ICML 2026. Preprint: [arXiv:2602.04856](https://arxiv.org/abs/2602.04856).

```bibtex
@InProceedings{pmlr-v306-tong26e,
  title = 	 {{C}o{T} is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning {LLM}s for Fake News Generation},
  author =       {Tong, Zhao and Gong, Chunlin and Zhang, Yiping and Shi, Haichao and Liu, Qiang and Xu, Xingcheng and Wu, Shu and Zhang, Xiao-Yu},
  booktitle = 	 {Proceedings of the 43rd International Conference on Machine Learning},
  pages = 	 {121562--121589},
  year = 	 {2026},
  editor = 	 {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume = 	 {306},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--11 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v306/main/assets/tong26e/tong26e.pdf},
  url = 	 {https://proceedings.mlr.press/v306/tong26e.html},
  abstract = 	 {From generating headlines to fabricating news, the Large Language Models (LLMs) are typically assessed by their final outputs, under the safety assumption that a refusal response signifies safe reasoning throughout the entire process. Challenging this assumption, our study reveals that during fake news generation, even when a model rejects a harmful request, its Chain-of-Thought (CoT) reasoning may still internally contain and propagate unsafe narratives. To analyze this phenomenon, we introduce a unified safety-analysis framework that systematically deconstructs CoT generation across model layers and evaluates the role of individual attention heads through Jacobian-based spectral metrics. Within this framework, we introduce three interpretable measures: stability, geometry, and energy to quantify how specific attention heads respond or embed deceptive reasoning patterns. Extensive experiments on multiple reasoning-oriented LLMs show that the generation risk rise significantly when the thinking mode is activated, where the critical routing decisions concentrated in only a few contiguous mid-depth layers. By precisely identifying the attention heads responsible for this divergence, our work challenges the assumption that refusal implies safety and provides a new understanding perspective for mitigating latent reasoning risks.}
}
```
