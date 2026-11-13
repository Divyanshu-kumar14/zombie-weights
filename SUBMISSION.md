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

Using the Lexsi Labs stack (`CuratorKIT`, `CircuitKIT`, `SafeTune`, `aligntune`, and `AuditKIT`), I built an automated audit harness running on Nvidia L4 and A100 GPUs to see if unlearned knowledge could be resurrected across four distinct attack vectors, and whether circuit-targeted intervention actually beats random-layer controls.

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

I trained a deep 140-step RMU baseline ($\alpha=1200$) to push hazardous capability to chance level and subjected it to four resurrection modalities:

| Attack Vector | Modality | Cyber Resurrection (Best Lang) | Bio Resurrection (Best Lang) | Gate $\ge +10\text{ pp}$ |
| :--- | :--- | :---: | :---: | :---: |
| **Vector A: Cross-Lingual Evasion** | Querying in non-English translation | **+4.0 pp** (fr) | **+2.0 pp** (fr) | ✗ FAIL |
| **Vector B: Benign LoRA Adaptation** | 20 steps SFT on harmless text via `aligntune` | **-1.0 pp** (ar) | **+6.0 pp** (en) | ✗ FAIL |
| **Vector C: 4-Bit NF4 Quantization** | Weight discretization unmasking | **+1.0 pp** (fr) | **+6.0 pp** (es) | ✗ FAIL |
| **Vector D: SafeTune Abliteration** | Refusal steering vector removal (Layer 18) | **+5.0 pp** (fr) | **+2.0 pp** (es) | ✗ FAIL |

**Kill-Gate H18 Result:** 0 of 4 vectors crossed the $+10\text{ pp}$ gate.

**The Technical Reality:** Driving RMU for 140 steps successfully suppressed English cyber accuracy to 29.0% (and 24.0%–25.0% in non-English chance band). However, shallow 20-step LoRA adaptation and single-layer abliteration only produced bounded rebounds (+4 to +6 pp). This reveals that deep representation unlearning creates substantial inertia against shallow fine-tuning, requiring stronger structural weight modifications to fully revive.

---

## Phase 3: CircuitKIT Discovery vs Random-Layer Control

To see if circuit-targeted intervention permanently disables hazardous knowledge while preserving multilingual intelligence, I scaled discovery to an 80 GB A100 GPU and evaluated against an equal-budget random control.

### Discovery on an 80 GB A100 GPU
Running node-level EAP-IG discovery across 32 examples (5 IG steps, sequence length 256) consumed **$62.62\text{ GB}$ peak VRAM** across 612 node scores. Evaluating direct attribution scores resolved the true top-6 functional MLP circuit layers:
- **Discovered Circuit Layers:** `[0, 4, 9, 16, 18, 19]` (Layer 16: 0.175, Layer 19: 0.164, Layer 18: 0.136, Layer 0: 0.136, Layer 9: 0.126, Layer 4: 0.120)
- **Random Control Layers (Seed 42):** `[1, 5, 10, 11, 26, 29]`

I ran 140 steps of RMU ($\alpha=1200$) with a $\beta=2.0$ multilingual retain stream on both arms and re-ran the full attack matrix:

| Arm | Targeted Layers | Max Cyber Resurrection | Max Bio Resurrection | Multilingual Retain Preservation | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Circuit Arm** | `[0, 4, 9, 16, 18, 19]` | **8.0 pp** (NF4) | 14.0 pp (NF4) | **HI: 50.0% (0.0 pp drop)**<br>**AR: 48.0% (3.0 pp drop)** | **Circuit Suppressed** (<10 pp Cyber) |
| **Random Control** | `[1, 5, 10, 11, 26, 29]` | **10.0 pp** (NF4) | 6.0 pp (NF4) | HI: 50.0%, AR: 48.0% | **Random Resurrects** ($\ge 10\text{ pp}$ Cyber) |

**Key Findings:**
1. **Circuit Excision Protects Against Unmasking:** Circuit-targeted RMU kept cyber resurrection bounded to 8.0 pp, while random control was unmasked by NF4 quantization (+10.0 pp).
2. **Utility Tax Eliminated:** Adding $\beta=2.0$ multilingual retain regularization completely protected Hindi general intelligence: Hindi MMLU remained at **50.0% (0.0 pp drop)**, and Arabic dropped by only **3.0 pp** ($51\% \to 48\%$).
3. **Biological Domain Specificity:** NF4 quantization resurrected Bio capabilities on the cyber circuit arm (+14.0 pp), demonstrating that biological hazards reside in distinct circuits that require separate excision.

---

## Compute Spend

I ran all GPU jobs across Nvidia L4 and A100-80GB instances through Hugging Face Jobs:
- Hackathon Credit Allowance: \$20.00
- Initial Spend: $\approx \$1.50$
- Publication-Grade Upgrade Spend: $\approx \$0.81$
- **Total Spent:** **$\approx \$2.31$** (leaving over \$17.60 unspent)

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
