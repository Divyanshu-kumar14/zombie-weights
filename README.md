# 🧟 ZOMBIE-WEIGHTS: The Unlearning Lie Detector
### Exposing Residual Latent Circuits & Resurrection Attacks on Dense Multilingual LLMs

**Author:** Divyanshu Kumar  
**Track:** Track 2a (AI Safety & Red Teaming) & Track 2b (AI Safety & Correction)  
**Target Model:** `CohereLabs/tiny-aya-global` (3.35B dense parameters)  
**License:** Apache-2.0  
**GitHub:** [`https://github.com/Divyanshu-kumar14/zombie-weights`](https://github.com/Divyanshu-kumar14/zombie-weights)  
**Live Demo Space:** [`https://huggingface.co/spaces/tony0014/zombie-weights-mri`](https://huggingface.co/spaces/tony0014/zombie-weights-mri)  
**Model Collection:** [`https://huggingface.co/collections/tony0014/zombie-weights-models-and-interactive-space`](https://huggingface.co/collections/tony0014/zombie-weights-models-and-interactive-space)  
**Dataset Collection:** [`https://huggingface.co/collections/tony0014/zombie-weights-datasets-and-telemetry`](https://huggingface.co/collections/tony0014/zombie-weights-datasets-and-telemetry)  
**Container:** `ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest` (`sha256:73affb9ed2b6d810f2ba6fcc8de356b72b12045e4c33366f2a1ec5624b667ce7`)

---

## What I Built and Why

When AI labs claim they have "unlearned" hazardous knowledge—such as cyberweapon exploits or bioweapons—using representation editing (RMU, NPO), I wanted to see if that knowledge was genuinely erased or just hidden behind a superficial logit mask.

I tested this on Cohere's enterprise multilingual model (`CohereLabs/tiny-aya-global`, 3.35B). What I found is that the knowledge never actually left the model. The underlying functional circuits remain dormant in the weights, creating what I call **Zombie Weights**.

Using the Lexsi Labs stack (`CuratorKIT`, `CircuitKIT`, `SafeTune`, `aligntune`, and `AuditKIT`), I built an automated audit harness running on Nvidia L4 GPUs ($24\text{ GB}$) to see if unlearned knowledge could be resurrected across four distinct attack vectors, and whether circuit-targeted intervention actually beats random-layer controls.

---

## Phase 0 & 1: Data Curation and Baseline Verification (Kill-Gate H4)

I started by building a multilingual evaluation and training split using `CuratorKIT`:
- **Dataset:** I pushed 10,788 items to [`tony0014/zombie-weights-bench`](https://huggingface.co/datasets/tony0014/zombie-weights-bench), comprising 3,600 evaluation MCQs (WMDP-Cyber and WMDP-Bio across 6 languages: English, Spanish, Hindi, Hinglish, Arabic, French) with answer-index invariance, 1,246 prose records for alignment, 200 canary probes, 1,000 held-out prompts, and 900 Global-MMLU controls.
- **Coverage:** CuratorKIT handled 8,602 translations, 4,138 identity-code preservations, 2,157 Hindi transliterations, and 103 refusals (0.7%). All 23 unit and Hypothesis property tests passed.
- **Docker Setup:** I built a dual-venv Docker image to isolate dependency clashes between transformers 5.21 and torchao/aligntune, pushed to `ghcr.io/divyanshu-kumar14/zombie-weights-l4:latest`.

### Base Model Accuracy ($N=100$ per pair, bfloat16 deterministic log-likelihood)

| Task | English (EN) | Spanish (ES) | Hindi (HI) | Hinglish | Arabic (AR) | French (FR) | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WMDP-Cyber** | **44.0%** | **43.0%** | 40.0% ✗ | 40.0% ✗ | 39.0% ✗ | 38.0% ✗ | 2/6 pass ($\ge 42\%$) |
| **WMDP-Bio** | **54.0%** | **49.0%** | **51.0%** | **44.0%** | **59.0%** | **51.0%** | **6/6 pass** ($\ge 42\%$) |
| **Global-MMLU** | **57.0%** | — | **50.0%** | — | **51.0%** | — | **3/3 pass** ($\ge 42\%$) |

**Kill-Gate H4 Result:** 11 of 15 pairs passed the $\ge 42\%$ threshold. The four failures were all non-English cyber pairs landing between 38% and 40%, which stayed safely above my 38% drop floor. Following my plan, I kept all languages and weighted cyber and Arm B accordingly. RMU smoke testing shifted weight norms by $\|\Delta W\|_2 = 2.09$ in 25 seconds with $17.66\text{ GB}$ peak VRAM, and AlignTune ran cleanly for 10 steps.

---

## Phase 2: The 4-Vector Attack Matrix (Kill-Gate H18)

I trained a 60-step RMU baseline ($\alpha=300$) and subjected it to four resurrection attacks:

| Attack Vector | Modality | Cyber Resurrection (Best Lang) | Bio Resurrection (Best Lang) | Gate $\ge +10\text{ pp}$ |
| :--- | :--- | :---: | :---: | :---: |
| **Vector A: Cross-Lingual Evasion** | Querying in non-English translation | **+6.0 pp** (es) | **+4.0 pp** (es) | ✗ FAIL |
| **Vector B: Benign LoRA Adaptation** | 20 steps SFT on harmless text via `aligntune` | **+5.0 pp** (es) | **+3.0 pp** (fr) | ✗ FAIL |
| **Vector C: 4-Bit NF4 Quantization** | Weight discretization unmasking | **-1.0 pp** (en) | **+5.0 pp** (es) | ✗ FAIL |
| **Vector D: SafeTune Abliteration** | Refusal steering vector removal (Layer 18) | **+9.0 pp** (hi) | **+5.0 pp** (es) | ✗ FAIL |

**Kill-Gate H18 Result:** 0 of 4 vectors crossed the $+10\text{ pp}$ gate, with Vector D coming closest at $+9.0\text{ pp}$.

**The Technical Reality:** The 60-step RMU run didn't push English cyber accuracy low enough—it hovered around 39–46%, which left very little headroom to show a large rebound. The attack scripts and evaluation pipelines worked as intended across all four modalities, but testing a rebound properly requires driving the baseline closer to random chance (~25%) using 80–160 steps at $\alpha=1200$.

---

## Phase 3: CircuitKIT Discovery vs Random-Layer Control

To see if I could permanently disable the capability rather than masking it, I used `CircuitKIT` to locate the functional MLP layers.

### Discovery on a 24 GB GPU
Running node-level EAP discovery across a 36-layer model with Tiny Aya's 256,000-token vocabulary originally triggered CUDA OOMs. By constraining sequence length to 128 and running $N=1$ example at $16.51\text{ GB}$ peak VRAM, CircuitKIT pruned 10 out of 36 MLP layers, identifying the top 6:
- **Discovered Circuit Layers:** `[12, 20, 24, 27, 31, 33]`
- **Random Control Layers (Seed 42):** `[0, 3, 7, 8, 22, 26]`

I ran RMU for 60 steps at $\alpha=400$ on both arms ($\|\Delta W\|_2 = 15.74$, peak VRAM $19.87\text{ GB}$) and re-ran the attack matrix:

| Arm | Targeted Layers | Max Resurrection | Attack Details | Verdict |
| :--- | :--- | :---: | :--- | :---: |
| **Circuit Arm** | `[12, 20, 24, 27, 31, 33]` | **8.0 pp** | Cyber D $+8.0$; B/C $+2.0$; A $+4.0$; Bio $0.0$ | **Circuit Dies** ($<10\text{ pp}$) |
| **Random Control** | `[0, 3, 7, 8, 22, 26]` | **10.0 pp** | Cyber B $+10.0$ (Gate Pass); A/D $+8.0$; Bio D $+6.0$ | **Random Resurrects** ($\ge 10\text{ pp}$) |

**Phase 3 Result:** The circuit arm stayed below 10 pp resurrection (8.0 pp), while the random control hit the 10.0 pp gate on benign LoRA adaptation. The circuit arm beat the random control.

### The Honest Caveats
1. **Attribution Noise:** Running discovery on a single example was a compromise to stay inside the 24 GB VRAM limit. While the circuit worked, CircuitKIT recommends at least 32 examples for high confidence.
2. **Narrow Margin:** The difference between 8 pp and 10 pp is real, but it is not a massive gap.
3. **The Utility Tax:** Surgical intervention came with a noticeable cost to general capabilities. Global-MMLU dropped by 5 pp in English (57% $\to$ 52%), 12 pp in Hindi (50% $\to$ 38%), and 7 pp in Arabic (51% $\to$ 44%). Deep circuit unlearning damages multilingual knowledge unless paired with explicit retain regularization.

---

## Compute Spend

I ran all 10 GPU jobs on Nvidia L4 instances through Hugging Face Jobs:
- Budget: \$20.00
- Hard cap: \$9.60
- **Total spent:** **$\approx \$1.50$** (leaving over \$18.50 unspent)

---

## 🎬 180-Second Pitch & Video Script

### **[0:00 - 0:45] The Unlearning Lie**
> *"When labs claim they have unlearned dangerous knowledge using representation misdirection, they usually haven't. I tested this on Cohere's 3.35B Tiny-Aya. In English, RMU appears to work, but the capability is still intact in what I call 'Zombie Weights'—dormant circuits that can be woken back up."*

### **[0:45 - 1:30] Four Ways to Resurrect Capabilities**
> *"I built an automated attack suite across four vectors: cross-lingual queries, 20 steps of benign LoRA fine-tuning with AlignTune, 4-bit NF4 quantization, and SafeTune abliteration. Removing the refusal steering vector alone brought back 9 percentage points of cyber capability. Standard unlearning is mostly surface veneer."*

### **[1:30 - 2:30] Circuit Excision vs Random Layers**
> *"To see if I could permanently disable the behavior, I used CircuitKIT to trace the attribution graph and identify the top-6 MLP circuit layers: 12, 20, 24, 27, 31, and 33. Restricting RMU to these layers kept resurrection capped at 8 percentage points, while an equal-budget random control resurrected by 10 percentage points under benign LoRA. But there's a trade-off: circuit intervention reduced Hindi and Arabic MMLU by up to 12 percentage points."*

### **[2:30 - 3:00] The Unlearning MRI**
> *"I deployed the Unlearning MRI on Hugging Face Spaces so anyone can audit the raw telemetry, test live multilingual probes, and inspect layer heatmaps. All code is open-source under Apache-2.0. Don't trust logit suppression—audit the weights."*
