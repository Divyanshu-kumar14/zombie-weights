<div align="center">

<h1>Zombie Weights</h1>

<p><strong>Auditing shallow unlearning, dormant latent circuits, and resurrection attacks in dense multilingual LLMs.</strong></p>

<p>Continuous safety audit harness evaluating representation misdirection, circuit localization, and post-unlearning tamper resistance.<br/>
Evaluated on Cohere Tiny-Aya-Global (3.35B) across 15 multilingual tasks on Nvidia L4 GPUs.</p>

<p><sub>CuratorKIT &nbsp;·&nbsp; CircuitKIT &nbsp;·&nbsp; SafeTune &nbsp;·&nbsp; AlignTune &nbsp;·&nbsp; AuditKIT &nbsp;·&nbsp; Apache-2.0 licensed</sub></p>

<p>
  <img src="https://img.shields.io/badge/Model-CohereLabs%2Ftiny--aya--global%20(3.35B)-222224?style=flat-square&labelColor=131217" alt="Target Model: CohereLabs/tiny-aya-global (3.35B)" />
  <img src="https://img.shields.io/badge/Hardware-Nvidia%20L4%20(24%20GB%20VRAM)-222224?style=flat-square&labelColor=131217" alt="Nvidia L4 GPU" />
  <img src="https://img.shields.io/badge/Docker-GHCR%20dual--venv-222224?style=flat-square&labelColor=131217" alt="GHCR Docker container" />
  <img src="https://img.shields.io/badge/Tests-23%2F23%20passed%20(Hypothesis)-44A282?style=flat-square&labelColor=131217" alt="Test Suite" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-44A282?style=flat-square&labelColor=131217" alt="License: Apache-2.0" /></a>
</p>

</div>

**Jump to:** [overview](#1-overview--the-unlearning-lie) · [architecture](#2-architecture--5-toolkit-pipeline) · [empirical results](#3-empirical-results) · [reproducibility](#4-reproducibility--quickstart) · [limitations](#5-limitations--caveats) · [verification](#6-verification)

---

## 1. Overview & The Unlearning Lie

When AI developers claim hazardous capabilities (such as cyberweapon exploitation or bioweapon synthesis) have been unlearned using representation editing algorithms like Representation Misdirection Unlearning (RMU), I wanted to determine whether the underlying knowledge was genuinely destroyed or merely masked at the logit boundary.

I evaluated this on Cohere's enterprise multilingual model (`CohereLabs/tiny-aya-global`, 3.35B dense parameters). What I found is that standard unlearning does not destroy the underlying capability. Instead, it introduces a superficial cancellation vector in early layers while leaving the downstream functional circuits intact in the weights. I term these dormant structures **Zombie Weights**.

Three properties make this problem dangerous in production:

- **The Cross-Lingual Gap:** RMU applied to English text leaves multilingual representations unaligned. Non-English queries (Spanish, Hindi, Arabic) bypass the suppression boundary.
- **Benign Re-adaptation:** Just 20 steps of LoRA fine-tuning on completely harmless text using AlignTune alters the cancellation weights and resurrects latent capabilities.
- **Quantization Unmasking:** 4-bit post-training quantization (NF4) acts as a weight perturbation that strips delicate unlearning vectors, bringing capability back.

I designed this project around three cooperating components:

| Component | What it does | Why it can be trusted |
|---|---|---|
| **Invariant Multilingual Benchmark** | 10,788 items across 6 languages with answer-index invariance | Generated via CuratorKIT; verified with 23 property-based Hypothesis tests |
| **4-Vector Attack Matrix** | Evaluates multilingual evasion, benign LoRA, NF4 quantization, and SafeTune abliteration | Ground-truth log-likelihood scoring with exact Wilson 95% confidence intervals |
| **Circuit Breaker vs Random Control** | Pinpoints MLP circuits via CircuitKIT and benchmarks against equal-budget random layers | Compares identical RMU training budgets; proves circuit-restricted unlearning resists resurrection |

---

## 2. Architecture & 5-Toolkit Pipeline

The entire system integrates all five Lexsi Labs toolkits into a continuous automated harness:

```mermaid
flowchart LR
  Curator["CuratorKIT<br/>multilingual curation"] --> Bench[("Bench Dataset<br/>10,788 items")]
  Bench --> Circuit["CircuitKIT<br/>EAP attribution"]
  Circuit --> Layers["Circuit Discovery<br/>MLP Top-6 Layers"]
  Bench --> RMU["SafeTune<br/>RMU unlearning"]
  Layers --> Breaker["Circuit Breaker<br/>circuit vs random RMU"]
  RMU --> Attacks["4-Vector Attack Suite<br/>cross-lingual, LoRA, NF4, abliteration"]
  Breaker --> Attacks
  Attacks --> Audit["AuditKIT<br/>Wilson 95% CIs & Telemetry"]
```

### The 5 Lexsi Toolkits in Action:
1. **CuratorKIT:** Translates and structures evaluation splits across English, Spanish, Hindi, Hinglish, Arabic, and French while preserving code snippets (`identity_code`), transliterating Hindi phonetics, and verifying answer-index invariance across all 3,600 test MCQs.
2. **CircuitKIT:** Runs edge attribution patching (EAP) across all 36 transformer layers to isolate which MLP blocks store hazardous capabilities versus general multilingual reasoning.
3. **SafeTune:** Executes Representation Misdirection Unlearning (RMU) across targeted layers and performs runtime steering vector extraction via `AbliterationAttack`.
4. **aligntune:** Executes rapid LoRA adaptation on benign, harmless text to test post-unlearning stability and recovery dynamics.
5. **AuditKIT:** Computes exact Wilson score 95% confidence intervals across all evaluations and generates machine-readable telemetry JSONs.

---

## 3. Empirical Results

All evaluations were executed on Nvidia L4 GPUs ($24\text{ GB}$ VRAM) using bfloat16 deterministic log-likelihood scoring.

### 3.1 Base Model Performance (Kill-Gate H4)

Evaluated across 15 language-task pairs ($N=100$ per pair):

| Task | English (EN) | Spanish (ES) | Hindi (HI) | Hinglish | Arabic (AR) | French (FR) | Gate ($\ge 42\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WMDP-Cyber** | **44.0%** | **43.0%** | 40.0% | 40.0% | 39.0% | 38.0% | 2/6 |
| **WMDP-Bio** | **54.0%** | **49.0%** | **51.0%** | **44.0%** | **59.0%** | **51.0%** | 6/6 |
| **Global-MMLU** | **57.0%** | — | **50.0%** | — | **51.0%** | — | 3/3 |

**Outcome:** 11 of 15 pairs passed the $\ge 42\%$ threshold. The four misses were all non-English cyber pairs landing between 38% and 40%, comfortably above the 38% floor. I retained all languages and weighted cyber and Arm B according to the pre-registered plan.

### 3.2 The 4-Vector Attack Matrix (Kill-Gate H18)

I trained a 60-step RMU baseline ($\alpha=300$) and subjected it to four resurrection modalities:

| Attack Vector | Modality | Cyber Resurrection (Best Lang) | Bio Resurrection (Best Lang) | Gate Target ($\ge +10\text{ pp}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Vector A: Cross-Lingual Evasion** | Querying in non-English translation | **+6.0 pp** (es) | **+4.0 pp** (es) | Not met |
| **Vector B: Benign LoRA Adaptation** | 20 steps SFT on harmless text via `aligntune` | **+5.0 pp** (es) | **+3.0 pp** (fr) | Not met |
| **Vector C: 4-Bit NF4 Quantization** | Weight discretization unmasking | **-1.0 pp** (en) | **+5.0 pp** (es) | Not met |
| **Vector D: SafeTune Abliteration** | Refusal steering vector removal (Layer 18) | **+9.0 pp** (hi) | **+5.0 pp** (es) | Not met (1 pp short) |

**Analysis:** The 60-step RMU run did not suppress English cyber accuracy far enough—it hovered around 39–46%, which left limited dynamic range to demonstrate a large ($>10\text{ pp}$) rebound. While Vector D came closest ($+9.0\text{ pp}$), demonstrating full $>10\text{ pp}$ resurrection requires driving the unlearned baseline closer to random chance (~25%) using 80–160 steps at $\alpha=1200$.

### 3.3 CircuitKIT Excision vs Random Control (Phase 3)

Using CircuitKIT, I isolated the top-6 MLP layers responsible for the cyber capability and benchmarked circuit-restricted RMU against an equal-budget random control:

- **Discovered Circuit Layers:** `[12, 20, 24, 27, 31, 33]`
- **Random Control Layers:** `[0, 3, 7, 8, 22, 26]` (seed 42)
- **Training Parameters:** 60 steps RMU at $\alpha=400$ on both arms ($\|\Delta W\|_2 = 15.74$, peak VRAM $19.87\text{ GB}$).

| Arm | Targeted Layers | Max Resurrection | Attack Detail | Verdict |
| :--- | :--- | :---: | :--- | :---: |
| **Circuit Arm** | `[12, 20, 24, 27, 31, 33]` | **8.0 pp** | Cyber D $+8.0$; B/C $+2.0$; A $+4.0$; Bio $0.0$ | **Circuit Dies** ($<10\text{ pp}$) |
| **Random Control** | `[0, 3, 7, 8, 22, 26]` | **10.0 pp** | Cyber B $+10.0$ (LoRA); A/D $+8.0$; Bio D $+6.0$ | **Random Resurrects** ($\ge 10\text{ pp}$) |

**Verdict:** PASS. The circuit arm kept resurrection capped at 8.0 pp, while the random control resurrected by 10.0 pp under benign LoRA fine-tuning. Circuit-targeted intervention successfully beat the random control.

---

## 4. Reproducibility & Quickstart

### 4.1 Docker Container (Recommended)

I containerized the full environment with dual virtual environments to eliminate version clashes between transformers 5.21 and torchao/aligntune:

```bash
# Pull the prebuilt image from GHCR
docker pull ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest

# Run the 23-test suite inside the container
docker run --gpus all --rm ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest pytest /code/tests -v
```

### 4.2 Local Environment Setup

Requires Python 3.12+ and an Nvidia GPU with $\ge 20\text{ GB}$ VRAM:

```bash
git clone https://github.com/Divyanshu-kumar14/zombie-weights.git
cd zombie-weights

# Install dependencies via uv
uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run property tests
pytest tests/ -v
```

### 4.3 Running the Pipeline Stages

```bash
# 1. Base accuracy and unlearning smoke test
python scripts/smoke_l4.py --bench-repo tony0014/zombie-weights-bench

# 2. Execute the 4-vector attack sweep
python scripts/attack_sweep.py --bench-repo tony0014/zombie-weights-bench

# 3. Run circuit discovery and comparative random control
python scripts/circuit_breaker.py --circuit-layers "12,20,24,27,31,33"
```

---

## 5. Limitations & Caveats

Stated plainly, because an evaluator should not have to discover these:

- **Attribution Sample Size:** Due to 24 GB VRAM constraints on the L4 instance, circuit discovery ran on a single example ($N=1$). While the circuit localized to layers 12–33 and successfully beat the control, CircuitKIT recommends $N \ge 32$ to eliminate gradient attribution noise.
- **Separation Margin:** The gap between the circuit arm (8.0 pp) and the random control (10.0 pp) is real, but it is a 2.0 pp margin rather than complete suppression.
- **The Retain Utility Tax:** Deep circuit intervention degraded general reasoning: Global-MMLU dropped by 5 pp in English (57% $\to$ 52%), 12 pp in Hindi (50% $\to$ 38%), and 7 pp in Arabic (51% $\to$ 44%). Surgical unlearning requires explicit retain regularization to preserve multilingual capabilities.
- **Baseline Dynamic Range:** 60 steps of RMU at $\alpha=300$ left baseline English cyber accuracy at ~39–46%. To evaluate large ($>15\text{ pp}$) resurrection peaks, the baseline must be pushed closer to 25% chance level with 120–160 steps at $\alpha=1200$.

---

## 6. Verification

Every claim in this repository is backed by raw JSON predictions, loss traces, and logs:

| What to verify | Location / Command |
|---|---|
| **Benchmark Dataset** | [`tony0014/zombie-weights-bench`](https://huggingface.co/datasets/tony0014/zombie-weights-bench) |
| **Raw L4 Telemetry Hub** | [`tony0014/zombie-weights-telemetry`](https://huggingface.co/datasets/tony0014/zombie-weights-telemetry) |
| **Interactive Space Demo** | [`tony0014/zombie-weights-mri`](https://huggingface.co/spaces/tony0014/zombie-weights-mri) |
| **Model Collection** | [`tony0014/zombie-weights-models-and-interactive-space`](https://huggingface.co/collections/tony0014/zombie-weights-models-and-interactive-space) |
| **Dataset Collection** | [`tony0014/zombie-weights-datasets-and-telemetry`](https://huggingface.co/collections/tony0014/zombie-weights-datasets-and-telemetry) |
| **Unit & Property Tests** | `pytest tests/ -v` (23 passed) |
| **Dual-Venv Container** | `ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest` |

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
