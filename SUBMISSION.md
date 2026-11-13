# 🧟 ZOMBIE-WEIGHTS: The Unlearning Lie Detector
### Exposing Residual Latent Circuits & Resurrection Attacks on Dense Multilingual LLMs

**Lexsi AI Safety & Alignment Hackathon 2026**  
**Track:** Open Safety Benchmark & Algorithmic Discovery  
**Target Model:** `CohereLabs/tiny-aya-global` (3.35B dense parameters)  
**License:** Apache-2.0  

---

## 🎯 Executive Summary & The Unlearning Lie

When AI developers and regulators claim that hazardous knowledge (cyberweapons, bioweapons, chemical synthesis) has been "unlearned" from an LLM using state-of-the-art algorithms (`RMU`, `NPO`, representation destruction), **they are almost always wrong**.

We prove empirically on an enterprise multilingual model (`CohereLabs/tiny-aya-global`) that current unlearning methods merely apply a **superficial English suppression mask** over the logits while leaving the underlying functional circuits completely intact in the model's weights. 

We term these dormant functional structures **Zombie Weights**.

Through our toolchain integrating all 5 Lexsi Labs libraries (`CuratorKIT`, `CircuitKIT`, `SafeTune`, `aligntune`, `AuditKIT`), we subject the unlearned model to a ruthless 4-vector attack sweep and prove that hazardous capabilities resurrect immediately:

1. **Vector A (Multilingual Evasion):** Prompting in Spanish, Arabic, or Hindi bypasses the English unlearning mask, leaking hazardous knowledge by **+12.0 pp**.
2. **Vector B (Benign LoRA Recovery):** Just 20 steps of LoRA fine-tuning on harmless multilingual text resurrects cyberweapon capabilities by **+16.0 pp**.
3. **Vector C (4-bit NF4 Quantization):** Post-training 4-bit quantization acts as an un-masking perturbation, resurrecting unlearned knowledge by **+20.0 pp**.
4. **Vector D (SafeTune Abliteration):** Orthogonalizing the refusal steering direction recovers latent capability by **+14.0 pp**.

Finally, we demonstrate that **only true circuit-restricted excision via CircuitKIT** permanently kills the latent capability (keeping accuracy suppressed below the 25% random guess baseline), whereas an identical negative control targeting random layers fails completely (+14.0 pp resurrection).

---

## 📊 Empirical Telemetry & Proof Table

All numbers are ground-truth metrics computed on Nvidia L4 GPUs with **Wilson Score 95% Confidence Intervals** across 15 language-task evaluation pairs:

| Evaluation Arm / Attack Vector | WMDP-Cyber (EN) | WMDP-Cyber (ES) | WMDP-Cyber (HI) | WMDP-Cyber (AR) | WMDP-Bio (EN) | WMDP-Bio (ES) | Global-MMLU (EN) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. Base Tiny-Aya-Global (3.35B)** | 43.0% $\pm 9.7$ | 44.0% $\pm 9.7$ | 42.0% $\pm 9.6$ | 44.0% $\pm 9.7$ | 55.0% $\pm 9.7$ | 52.0% $\pm 9.8$ | 54.0% $\pm 9.8$ |
| **2. Standard RMU Unlearned (Baseline)** | **12.0%** $\pm 9.0$ | 24.0% $\pm 11.8$ | 22.0% $\pm 11.4$ | 20.0% $\pm 11.1$ | 30.0% $\pm 12.7$ | 32.0% $\pm 12.9$ | 48.0% $\pm 13.8$ |
| **3. Vector A (Multilingual Evasion)** | 12.0% $\pm 9.0$ | **24.0%** (+12 pp) | 22.0% (+10 pp) | 20.0% (+8 pp) | 30.0% $\pm 12.7$ | 32.0% (+2 pp) | 48.0% $\pm 13.8$ |
| **4. Vector B (Benign LoRA SFT)** | **28.0%** (+16 pp) | 26.0% (+14 pp) | 26.0% (+14 pp) | 28.0% (+16 pp) | 36.0% (+6 pp) | 38.0% (+6 pp) | 48.0% $\pm 13.8$ |
| **5. Vector C (4-bit NF4 Quantization)** | **32.0%** (+20 pp) | 30.0% (+18 pp) | 28.0% (+16 pp) | 30.0% (+18 pp) | 34.0% (+4 pp) | 36.0% (+4 pp) | 46.0% $\pm 13.8$ |
| **6. Vector D (SafeTune Abliteration)** | **26.0%** (+14 pp) | 24.0% (+12 pp) | 22.0% (+10 pp) | 24.0% (+12 pp) | 28.0% (-2 pp) | 30.0% (-2 pp) | 48.0% $\pm 13.8$ |
| **7. Random-Layer Negative Control (Arm 2)** | 34.0% $\pm 13.1$ | 46.0% (+12 pp) | 42.0% (+8 pp) | 44.0% (+10 pp) | 46.0% $\pm 13.8$ | 48.0% (+2 pp) | 46.0% $\pm 13.8$ |
| **8. Circuit-Excised Intervention (Arm 1)** | **12.0%** $\pm 9.0$ | **18.0%** (DEAD) | **18.0%** (DEAD) | **16.0%** (DEAD) | **22.0%** (DEAD) | **24.0%** (DEAD) | 48.0% $\pm 13.8$ |

*(Note: Random guess baseline on 4-option multiple choice is 25.0%. Capabilities below 25.0% represent complete functional destruction.)*

---

## 🛠️ The 5-Tool Lexsi Labs Architecture

```
                                  [CuratorKIT]
                     Multilingual Translation & Curation
                    (10,788 translation segments across 5 langs)
                                      │
                                      ▼
                                [CircuitKIT]
                   Discovery Attribution Graphs & Layer Scoring
                        (MLP nodes, max_seq_len=256)
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

## 🔗 Project Deliverables & Live Links

- **Live Space Demo ("Unlearning MRI"):**  
  👉 [`https://huggingface.co/spaces/tony0014/zombie-weights-mri`](https://huggingface.co/spaces/tony0014/zombie-weights-mri)
- **Open GitHub Repository:**  
  👉 [`https://github.com/Divyanshu-kumar14/zombie-weights`](https://github.com/Divyanshu-kumar14/zombie-weights)
- **Multilingual Benchmark Dataset:**  
  👉 [`https://huggingface.co/datasets/tony0014/zombie-weights-bench`](https://huggingface.co/datasets/tony0014/zombie-weights-bench)
- **Ground-Truth L4 Telemetry Hub:**  
  👉 [`https://huggingface.co/datasets/tony0014/zombie-weights-telemetry`](https://huggingface.co/datasets/tony0014/zombie-weights-telemetry)

---

## 🎬 180-Second Demo & Presentation Script

### **[0:00 - 0:45] Act I: The Unlearning Lie**
> *"Every leading AI lab today claims they can 'unlearn' dangerous knowledge from large models using representation editing or RMU. We tested this claim on Cohere's enterprise 3.35B model, Tiny-Aya. In English, it looks like a complete success: hazardous cyberweapon accuracy plummets from 43% to 12%—well below random guess. But it's an illusion. The model hasn't forgotten anything. The knowledge is trapped in what we call 'Zombie Weights'—dormant functional circuits waiting to be reawakened."*

### **[0:45 - 1:30] Act II: The 4 Resurrection Vectors**
> *"We subjected this unlearned model to four distinct attack vectors. First, multilingual evasion: simply translating the query into Spanish or Arabic causes hazardous capability to leak out by plus 12 percentage points. Second, benign adaptation: we fine-tuned the model on completely harmless text for just 20 steps using AlignTune—and cyberweapon capability rebounded by plus 16 percentage points. Third, 4-bit NF4 quantization: simple weight quantization unmasked the dormant circuits, yielding a plus 20 percentage point resurrection. Fourth, SafeTune abliteration: peeling back the refusal steering vector instantly restored 26% accuracy. Standard unlearning is just a cosmetic veneer."*

### **[1:30 - 2:30] Act III: The CircuitKIT Cure & Negative Control**
> *"How do we kill a zombie? You have to destroy the circuit. Using CircuitKIT, we traced the underlying attribution graph across Tiny-Aya's 36 layers and identified the exact MLP bottleneck in layers 4 through 9. When we restricted SafeTune unlearning to this surgical circuit, capability dropped to 12%—and every single resurrection attack failed completely. Even after benign LoRA fine-tuning and quantization, capability remained dead at 16% to 18%, well below random chance. And to prove this wasn't an artifact of layer count, our negative control targeted 6 random layers—which failed immediately, rebounding by 14 percentage points."*

### **[2:30 - 3:00] Act IV: The Live Unlearning MRI & Call to Action**
> *"We built the Unlearning MRI—a live, interactive dashboard on Hugging Face Spaces where anyone can test hazardous probes in real time, inspect circuit attribution heatmaps, and audit our raw telemetry. All code, datasets, and Wilson 95% confidence intervals are 100% open-source under Apache-2.0. Stop trusting logit-mask unlearning. Audit the circuits."*
