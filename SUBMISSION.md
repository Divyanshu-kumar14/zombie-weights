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
| **Vector A: Cross-Lingual Evasion** | Querying in non-English translation | **+1.0 pp** (fr) | **+2.0 pp** (fr) | ✗ FAIL |
| **Vector B: Benign LoRA Adaptation** | 20 steps SFT on harmless text via `aligntune` | **+3.0 pp** (fr) | **+4.0 pp** (en) | ✗ FAIL |
| **Vector C: 4-Bit NF4 Quantization** | Weight discretization unmasking | **-3.0 pp** (fr) | **+3.0 pp** (es) | ✗ FAIL |
| **Vector D: SafeTune Abliteration** | Refusal steering vector removal (Layer 18) | **+4.0 pp** (es) | **-1.0 pp** (en) | ✗ FAIL |

**Kill-Gate H18 Result:** 0 of 4 vectors crossed the $+10\text{ pp}$ gate.

**The Technical Reality:** Driving RMU for 140 steps successfully suppressed English cyber accuracy to 29.0% (and 24.0%–25.0% in non-English chance band). However, shallow 20-step LoRA adaptation and single-layer abliteration only produced bounded rebounds (+4 to +6 pp). This reveals that deep representation unlearning creates substantial inertia against shallow fine-tuning, requiring stronger structural weight modifications to fully revive.

---

## Phase 3: CircuitKIT Discovery vs Random-Layer Control

To see if circuit-targeted intervention permanently disables hazardous knowledge while preserving multilingual intelligence, I scaled discovery to an 80 GB A100 GPU and evaluated against an equal-budget random control.

### Discovery on an 80 GB A100 GPU
Running node-level EAP-IG discovery across 32 examples (5 IG steps, sequence length 256) consumed **$62.62\text{ GB}$ peak VRAM** across 612 node scores. Evaluating direct attribution scores resolved the true top-6 functional MLP circuit layers:
- **Discovered Circuit Layers:** `[0, 4, 9, 16, 18, 19]` (Layer 16: 0.175, Layer 19: 0.164, Layer 18: 0.136, Layer 0: 0.136, Layer 9: 0.126, Layer 4: 0.120)
- **Random Control Layers (Seed 42):** `[1, 5, 10, 11, 26, 29]`

I ran 140 steps of RMU at $\alpha=1200$, $\beta=0$, $N=50$ per language-task cell on both arms and re-ran the full attack matrix:

| Arm | Targeted Layers | Baseline Cyber (EN) | Max Cyber Resurrection | Max Bio Resurrection | Global-MMLU After (EN/HI/AR) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Circuit Arm** | `[0, 4, 9, 16, 18, 19]` | **22.0%** | **+16.0 pp** (LoRA, es) | +4.0 pp (hi) | 20.0 / 20.0 / 20.0% | FAIL |
| **Random Control** | `[1, 5, 10, 11, 26, 29]` | **22.0%** | **+10.0 pp** (LoRA hi / NF4 hinglish) | +4.0 pp (hi) | 18.0 / 20.0 / 24.0% | FAIL |

**Pre-registered verdict (all four criteria):**

| Criterion | Threshold | Measured | Result |
| :--- | :--- | :---: | :---: |
| Circuit resurrection | $< 5$ pp | **16.0 pp** | FAIL |
| Random resurrection | $\ge 15$ pp | **10.0 pp** | FAIL |
| Circuit beats random | — | 16.0 vs 10.0 pp | FAIL |
| Global-MMLU drop | $\le 3$ pp | **37.0 pp EN** | FAIL |

**Key Findings:**
1. **The circuit hypothesis did not replicate.** Restricting RMU to the discovered early-layer MLP circuit suppressed English cyber to 22.0% (the 20%–25% chance band), but benign LoRA resurrected **+16.0 pp** — *more* than the random control's +10.0 pp. Circuit-targeted unlearning is not more attack-resistant at this scale.
2. **The utility tax is the binding constraint, not resurrection.** The same 140-step, $\alpha=1200$ run collapsed Global-MMLU from 57/50/51% to 20/20/20% (EN/HI/AR). Every configuration that protected utility also prevented forgetting: adding any competing retain term turns the weight trajectory into a near-random walk, $\|\Delta W\|_2$ collapses, the model stays functional, and cyber accuracy returns to 38%–44%.
3. **Why $\beta$ cannot buy both.** RMU's forget term pushes activations toward a random direction while the retain/utility terms pull them back. Under AdamW's normalized step size, $\|\Delta W\|_2$ drops from $\approx 28.7$ at $\beta=0$ to $\approx 4.0$–$4.4$ at $\beta>0$, so the model stays functional and cyber accuracy returns to 38%–44%.

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
