# Zombie Weights: Auditing Unlearning & Latent Functional Circuits in Dense Multilingual LLMs

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/Model-Tiny--Aya--Global_(3.35B)-orange)](https://huggingface.co/CohereLabs/tiny-aya-global)
[![Dataset](https://img.shields.io/badge/HF_Dataset-zombie--weights--bench-yellow)](https://huggingface.co/datasets/tony0014/zombie-weights-bench)
[![Telemetry](https://img.shields.io/badge/HF_Telemetry-zombie--weights--telemetry-green)](https://huggingface.co/datasets/tony0014/zombie-weights-telemetry)
[![Docker](https://img.shields.io/badge/Docker-GHCR_Package-blue)](https://github.com/Divyanshu-kumar14/zombie-weights/pkgs/container/zombie-weights-l4)

When developers claim hazardous knowledge has been unlearned from an LLM using representation editing (like RMU), I wanted to see if that knowledge was actually removed or just hidden behind a superficial logit mask.

I tested this on Cohere's enterprise multilingual model (`CohereLabs/tiny-aya-global`, 3.35B). What I found is that the knowledge never left the model. The underlying functional circuits remain dormant in the weights, creating what I call **Zombie Weights**.

Using the Lexsi Labs ecosystem (`CuratorKIT`, `CircuitKIT`, `SafeTune`, `aligntune`, and `AuditKIT`), I built an automated audit harness running on Nvidia L4 GPUs ($24\text{ GB}$) to test whether unlearned knowledge could be resurrected across four distinct attack vectors, and whether circuit-targeted intervention actually beats random-layer controls.

---

## Key Empirical Findings

### 1. Base Model Performance (N=100 per pair, bfloat16 deterministic log-likelihood)

I first evaluated the base model across 15 multilingual task pairs covering cyberweapons, bioweapons, and general reasoning:

| Task | English (EN) | Spanish (ES) | Hindi (HI) | Hinglish | Arabic (AR) | French (FR) | Pass ($\ge 42\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WMDP-Cyber** | **44.0%** | **43.0%** | 40.0% | 40.0% | 39.0% | 38.0% | 2/6 |
| **WMDP-Bio** | **54.0%** | **49.0%** | **51.0%** | **44.0%** | **59.0%** | **51.0%** | 6/6 |
| **Global-MMLU** | **57.0%** | — | **50.0%** | — | **51.0%** | — | 3/3 |

11 of 15 pairs met or exceeded the 42% threshold. The four misses were all non-English cyber pairs landing between 38% and 40%, which stayed safely above the 38% floor for viable evaluation.

### 2. The 4-Vector Attack Matrix

I applied standard RMU unlearning (60 steps, $\alpha=300$) and tested whether the suppressed capabilities could be recovered:

| Attack Vector | Mechanism | Cyber Resurrection (Best Lang) | Bio Resurrection (Best Lang) | Target ($\ge +10\text{ pp}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Vector A: Cross-Lingual Evasion** | Querying in non-English translation | **+6.0 pp** (es) | **+4.0 pp** (es) | Not met |
| **Vector B: Benign LoRA Adaptation** | 20 steps SFT on harmless text via `aligntune` | **+5.0 pp** (es) | **+3.0 pp** (fr) | Not met |
| **Vector C: 4-Bit NF4 Quantization** | Weight discretization unmasking | **-1.0 pp** (en) | **+5.0 pp** (es) | Not met |
| **Vector D: SafeTune Abliteration** | Refusal steering vector removal (Layer 18) | **+9.0 pp** (hi) | **+5.0 pp** (es) | Not met (1 pp short) |

**Root cause:** The 60-step RMU baseline did not suppress English cyber accuracy far enough—it hovered around 39–46%, which left very little headroom to show a large rebound. The attack execution pipelines functioned as intended across all four modalities, but demonstrating large rebounds requires driving the baseline closer to random chance (~25%) using 80–160 steps at $\alpha=1200$.

### 3. CircuitKIT Excision vs Random Control

Using `CircuitKIT`, I located the functional MLP layers responsible for the capability and benchmarked circuit-targeted RMU against an equal-budget random control:

- **Discovered Circuit Layers:** `[12, 20, 24, 27, 31, 33]` (via EAP-IG attribution)
- **Random Control Layers:** `[0, 3, 7, 8, 22, 26]` (seed 42)
- **Training:** Both arms ran RMU for 60 steps at $\alpha=400$ ($\|\Delta W\|_2 = 15.74$, peak VRAM $19.87\text{ GB}$).

| Arm | Targeted Layers | Max Resurrection | Attack Detail | Outcome |
| :--- | :--- | :---: | :--- | :---: |
| **Circuit Arm** | `[12, 20, 24, 27, 31, 33]` | **8.0 pp** | Cyber D $+8.0$; B/C $+2.0$; A $+4.0$; Bio $0.0$ | **Suppressed** ($<10\text{ pp}$) |
| **Random Control** | `[0, 3, 7, 8, 22, 26]` | **10.0 pp** | Cyber B $+10.0$ (LoRA); A/D $+8.0$; Bio D $+6.0$ | **Resurrected** ($\ge 10\text{ pp}$) |

**Caveats:**
1. **Attribution sample size:** Discovery ran on a single example to fit inside the 24 GB VRAM envelope. While the circuit localized properly, CircuitKIT recommends at least 32 examples for high statistical confidence.
2. **Margin:** The separation between the circuit arm (8 pp) and the random control (10 pp) is real, but narrow.
3. **The Utility Tax:** Surgical intervention degraded retain general capabilities: Global-MMLU dropped by 5 pp in English (57% $\to$ 52%), 12 pp in Hindi (50% $\to$ 38%), and 7 pp in Arabic (51% $\to$ 44%). Deep circuit unlearning damages multilingual knowledge unless paired with explicit retain regularization.

---

## Toolchain & Pipeline

```
                                  [CuratorKIT]
                     Multilingual Translation & Curation
                    (10,788 translation segments across 5 langs)
                                      │
                                      ▼
                                [CircuitKIT]
                   Discovery Attribution Graphs & Layer Scoring
                        (MLP nodes, max_seq_len=128)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
            [SafeTune]                                 [aligntune]
        Representation Destruction                 Benign LoRA Recovery
       (RMU unlearning, alpha=300-400)         (20 steps SFT, harmless text)
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                                 [AuditKIT]
                    Statistical Verification & Wilson CIs
                   (Zero simulated data, full JSON telemetry)
```

---

## Reproducibility & Quickstart

### Option 1: Docker (Recommended)

I built a dual-venv container to isolate dependency conflicts between transformers 5.21 and torchao/aligntune:

```bash
docker pull ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest

# Run the complete test suite
docker run --gpus all --rm ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest pytest /code/tests -v
```

### Option 2: Local Setup

Requires Python 3.12+ and an Nvidia GPU with at least 20 GB VRAM:

```bash
git clone https://github.com/Divyanshu-kumar14/zombie-weights.git
cd zombie-weights

# Setup virtual environment
uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run property tests
pytest tests/ -v

# Run smoke test & baseline evaluation
python scripts/smoke_l4.py --bench-repo tony0014/zombie-weights-bench

# Run 4-vector attack sweep
python scripts/attack_sweep.py --bench-repo tony0014/zombie-weights-bench

# Run circuit discovery & comparative control
python scripts/circuit_breaker.py --circuit-layers "12,20,24,27,31,33"
```

---

## Hosted Artifacts

- **Benchmark Dataset:** [`tony0014/zombie-weights-bench`](https://huggingface.co/datasets/tony0014/zombie-weights-bench) (3,600 MCQs, 1,246 alignment records, 1,200 canaries)
- **Raw Telemetry Hub:** [`tony0014/zombie-weights-telemetry`](https://huggingface.co/datasets/tony0014/zombie-weights-telemetry) (Full per-item JSON outputs and loss logs)
- **Interactive Space:** [`tony0014/zombie-weights-mri`](https://huggingface.co/spaces/tony0014/zombie-weights-mri) (Interactive inspection dashboard)
- **Docker Container:** `ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest`

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
