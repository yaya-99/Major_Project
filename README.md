# Grounded, Not Just Attending: A Targeted Consistency Objective for Attention-Correct, Semantically Incorrect Image Captioning Errors

### Reducing semantic grounding failures in vision–language image captioning models

---

## 📌 Overview

This repository contains the implementation and evaluation of our B.Tech Major Project, which extends the findings of our Minor Project on perceptual errors in image captioning systems.

The Minor Project showed an important failure pattern:

> **A model can attend to a visually relevant region while still producing semantically incorrect content.**

The Major Project investigates whether this failure can be reduced by explicitly encouraging alignment between:

- **contextual token representations** produced by the captioning model, and
- **visual-semantic representations** of the corresponding image regions.

Instead of directly modifying attention maps, the proposed method introduces a **targeted consistency objective** over eligible semantic tokens.

The central research question is:

> **Can targeted representation-level consistency reduce semantic grounding errors even when the model's perceptual attention is already appropriately directed?**

---

## 🎯 Research Objective

The project aims to improve semantic grounding in image captioning models by introducing an additional consistency objective alongside the standard captioning loss.

For an eligible caption token:

1. A contextual decoder representation is extracted under teacher forcing.
2. The corresponding visual region is represented using a frozen visual-semantic encoder.
3. The token representation is mapped into a shared representation space using a trainable projection.
4. The visual representation is mapped using a frozen projection.
5. A contrastive consistency objective encourages the correct token–region pair to be more similar than hard negatives from the same image.

The overall objective is:

\[
L_{\text{total}}
=
L_{\text{caption}}
+
\lambda L_{\text{consistency}}
\]

where:

- \(L_{\text{caption}}\) is the standard language-modeling loss.
- \(L_{\text{consistency}}\) is the targeted token–region consistency loss.
- \(\lambda\) controls the contribution of the consistency objective.

---

## 🔬 Research Hypothesis

We hypothesize that:

> **Explicit representation-level consistency between eligible semantic tokens and their corresponding visual regions will reduce semantic grounding errors compared with standard captioning fine-tuning alone.**

The intervention is specifically designed for cases where perceptual attention may already be reasonable but the generated semantic content is incorrect.

---

## ❓ Research Questions

### RQ1 — Semantic grounding

Does targeted token–region consistency reduce semantic hallucination and grounding errors in image captions?

### RQ2 — Mechanism

Does improving representation-level consistency provide evidence of improved grounding beyond standard captioning fine-tuning?

### RQ3 — Attention vs. semantics

Can a model exhibit appropriate visual attention while still producing semantically incorrect content, and does the proposed objective reduce this failure mode?

### RQ4 — Evaluation

Do improvements in semantic-grounding metrics occur without requiring improvements to be inferred solely from conventional caption-reference metrics?

---

## 🧠 Motivation from the Minor Project

The Major builds directly on observations from the Minor Project.

The Minor investigated:

- perceptual error patterns,
- hallucination and semantic errors,
- token-level cross-attention,
- human saliency alignment,
- and cross-model error behavior.

The attention analysis showed that model attention can be directed toward relevant visual regions even when the resulting semantic description is incorrect.

This motivates a different intervention:

> Rather than attempting to force attention toward a region, the Major explicitly encourages the **semantic representation of the generated content** to remain consistent with the corresponding visual evidence.

Attention analysis therefore remains an important diagnostic component, but it is **not the training objective**.

---

## 🏗️ Proposed Architecture

The Major uses a single full vision–language captioning backbone and introduces lightweight trainable components around it.

### Training pipeline

```text
                    ┌─────────────────────┐
                    │       Image         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    BLIP Vision      │
                    │      Encoder        │
                    └──────────┬──────────┘
                               │
                               ▼
                 Frozen visual-semantic target
                               │
                               ▼
                    Frozen visual projection
                               │
                               ▼
                              f_v


Ground-truth caption
        │
        ▼
┌──────────────────────────────┐
│ BLIP Decoder                 │
│ Teacher-forced forward pass  │
└──────────────┬───────────────┘
               │
               ▼
Contextual token hidden states
               │
               ▼
     Trainable token projection
               │
               ▼
              f_s

             f_s ↔ f_v
                │
                ▼
      Consistency / InfoNCE Loss

Total Loss:
L_total = L_caption + λ L_consistency
```

---

## 🔗 Token–Region Consistency

The consistency objective is applied selectively rather than to every generated token.

A token is considered eligible using the project's region-selection procedure, including:

- concrete noun part-of-speech categories,
- detector confidence,
- regional uniqueness.

Eligible tokens receive both:

```text
caption loss
+
consistency loss
```

Ineligible tokens still participate in the normal captioning objective:

```text
caption loss only
```

This prevents the consistency objective from being applied indiscriminately to function words or tokens without a reliable visual grounding target.

---

## 🧲 Hard Negatives

The consistency objective uses **same-image hard negatives**.

For an eligible token and its corresponding visual region:

```text
Positive:
token ↔ correct region

Negative:
token ↔ incorrect region from the same image
```

This creates a more targeted learning signal than simply contrasting examples from unrelated images.

The objective encourages the representation of a semantic token to be closer to its correct visual evidence than to competing regions within the same scene.

---

## 🧊 Frozen and Trainable Components

The Major deliberately limits the number of trainable components.

### Frozen

- pretrained BLIP backbone
- visual-semantic projection
- visual encoder used for the grounding target

### Trainable

- LoRA adapters applied to the selected BLIP modules
- token projection used to map contextual decoder representations into the shared consistency space

This keeps the intervention lightweight while allowing the language/decoder representation to adapt to the grounding objective.

---

## 🧪 Experimental Conditions

The Major compares the following conditions:

### 1. Frozen BLIP

The original pretrained model without additional fine-tuning.

This provides the pretrained baseline.

### 2. LoRA Captioning Baseline

BLIP is adapted using LoRA with the standard captioning objective:

\[
L = L_{\text{caption}}
\]

This isolates the effect of ordinary parameter-efficient fine-tuning.

### 3. LoRA + Consistency

BLIP is adapted using LoRA with:

\[
L =
L_{\text{caption}}
+
\lambda L_{\text{consistency}}
\]

This is the proposed method.

### Comparison

```text
                    ┌──────────────────┐
                    │   Frozen BLIP    │
                    └────────┬─────────┘
                             │
                             ▼
                    baseline performance


                    ┌──────────────────┐
                    │   BLIP + LoRA   │
                    │ Caption Loss     │
                    └────────┬─────────┘
                             │
                             ▼
                    fine-tuning baseline


                    ┌────────────────────────┐
                    │     BLIP + LoRA        │
                    │ Caption + Consistency  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                         Proposed method
```

---

## 👁️ Attention Diagnostic Pipeline

Attention analysis is retained from the Minor Project as a **diagnostic mechanism**, not as the training target.

The diagnostic pipeline is:

```text
Image
  ↓
BLIP caption generation
  ↓
Generated token
  ↓
Decoder cross-attention
  ↓
Head/layer aggregation
  ↓
Spatial attention representation
  ↓
Optional human-saliency comparison
  ↓
JS / KL divergence
```

This analysis asks:

> **Where is the model looking while generating a token?**

It does not directly answer:

> **Is the generated semantic content correct?**

The distinction is fundamental to the Major.

See:

`docs/attention_pipeline.md`

for the detailed diagnostic pipeline.

---

## 📊 Evaluation

The Major evaluates both conventional caption quality and semantic grounding.

### Conventional caption metrics

- BLEU-1
- BLEU-4
- CIDEr
- METEOR
- ROUGE-L
- SPICE

These metrics measure agreement with reference captions but are **not sufficient on their own to establish semantic grounding**.

### Semantic-grounding / hallucination metrics

The primary evaluation includes:

- **CHAIR-i**
- **CHAIR-s**
- **POPE**
- **ALOHa**

These metrics are used to evaluate hallucination and semantic grounding more directly.

### Diagnostic evidence

Additional evidence may include:

- token-level attention maps,
- attention–saliency divergence,
- token–region similarity,
- representation-level consistency,
- qualitative examples.

---

## ⚠️ Why Conventional Caption Metrics Are Not Enough

A caption can receive a strong reference-based score while still containing a semantically incorrect object, attribute, or relation.

Therefore:

```text
Caption similarity ≠ semantic correctness
```

The Major consequently treats BLEU, CIDEr, METEOR, ROUGE-L, and similar metrics as supporting evidence rather than as the sole measure of success.

The central evaluation focuses on whether the proposed method reduces semantic-grounding failures.

---

## 🧪 Dataset

The primary experimental dataset is:

**MS COCO**

The Major uses COCO image-caption data for model training, validation, and evaluation.

The Minor project's manually annotated failure corpus is retained as a **diagnostic resource**, not as the primary training dataset or headline evaluation set.

---

## 🔬 Experimental Design

The experimental workflow is:

```text
MS COCO
   │
   ├── Dataset preparation
   │
   ├── Baseline inference
   │
   ├── BLIP + LoRA training
   │
   ├── BLIP + LoRA + consistency training
   │
   ├── Caption generation
   │
   ├── Semantic-grounding evaluation
   │
   ├── Attention diagnostics
   │
   └── Statistical comparison
```

All experimental conditions should use the same evaluation protocol and prediction schema so that differences can be attributed to the training intervention rather than inconsistent evaluation procedures.

---

## 📁 Repository Structure

```text
.
├── README.md
├── requirements.txt
├── config.py
│
├── data/
│   ├── annotations/
│   └── ...
│
├── docs/
│   └── attention_pipeline.md
│
├── notebooks/
│   ├── 01_setup_and_dataset.ipynb
│   ├── 02_blip_architecture_and_generation.ipynb
│   ├── 03_evaluation_metrics.ipynb
│   ├── ...
│
├── src/
│   ├── ...
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   ├── checkpoints/
│   └── figures/
│
└── results/
    ├── tables/
    └── figures/
```

The repository structure may evolve as the training and evaluation pipeline is completed.

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

The Major uses the following primary technologies:

- PyTorch
- Hugging Face Transformers
- PEFT
- OpenCLIP
- torchvision
- pycocotools
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib
- seaborn

---

## 🚀 Experimental Workflow

The project is developed in stages.

### 1. Dataset setup

```text
notebooks/01_setup_and_dataset.ipynb
```

Responsible for:

- COCO dataset discovery,
- annotation loading,
- image/annotation validation,
- reproducibility setup,
- dataset manifest generation.

### 2. BLIP architecture and generation

```text
notebooks/02_blip_architecture_and_generation.ipynb
```

Responsible for:

- BLIP architecture probing,
- hidden-state inspection,
- cross-attention inspection,
- token-position verification,
- LoRA module discovery,
- baseline caption generation.

### 3. Evaluation

The evaluation notebooks compute conventional caption metrics and semantic-grounding metrics for standardized predictions.

### 4. Training

The training pipeline implements:

- teacher-forced contextual token extraction,
- region eligibility,
- visual representation extraction,
- token projection,
- frozen visual projection,
- LoRA adaptation,
- contrastive consistency loss,
- combined caption + consistency training.

### 5. Analysis

The final analysis compares the experimental conditions and combines:

- semantic-grounding metrics,
- conventional caption metrics,
- attention diagnostics,
- qualitative examples,
- statistical analysis.

---

## 📌 Reproducibility

The project uses explicit configuration and reproducibility controls.

Important experimental settings include:

- random seed,
- dataset split,
- model checkpoint,
- visual encoder checkpoint,
- LoRA configuration,
- consistency-loss weight,
- contrastive temperature,
- region eligibility thresholds,
- generation configuration.

The exact values used for a reported experiment should be recorded in the experiment configuration/manifest.

---

## 📈 Expected Contribution

The Major does not claim that attention itself is the cause of semantic errors.

Instead, it investigates whether a model that already possesses useful visual attention can be encouraged to produce more semantically grounded representations through an explicit consistency objective.

The intended contribution is therefore:

> **A targeted, parameter-efficient consistency intervention for reducing semantic grounding failures in image captioning.**

The work connects three levels of analysis:

```text
Perceptual focus
      ↓
Attention diagnostics

Representation-level grounding
      ↓
Token–region consistency

Semantic correctness
      ↓
Grounding / hallucination evaluation
```

---

## ⚖️ Ethics & Data Usage

- MS COCO and SALICON source datasets are not redistributed.
- Dataset licenses and terms of use should be respected.
- Only permitted derived metadata, annotations, and experimental artifacts should be included in the repository.
- Human annotation data should be handled according to the applicable project and dataset requirements.
- The project studies model behavior and semantic grounding; it does not claim to model or reproduce human cognition.

---

## 📌 What This Project Is NOT

This project is:

- not a leaderboard submission,
- not a general-purpose image captioning system,
- not an attempt to maximize conventional captioning metrics alone,
- not an attention-map optimization method,
- not a human cognition model.

It is a research investigation into whether **targeted representation-level consistency can reduce semantic grounding failures in vision–language image captioning models**.

---

## 👥 Contributors

### Abhidhey Singh

- Major project architecture and research direction
- BLIP integration and model development
- Baseline and LoRA training
- Token and visual representation integration
- Consistency objective and training pipeline
- Experimental analysis and interpretation
- Technical documentation and repository integration

### Sneha Mishra

- Dataset and annotation pipeline
- Semantic-grounding and hallucination evaluation
- Evaluation metric implementation
- Error categorization and qualitative analysis
- Statistical analysis and experimental comparison
- Human evaluation support

### Keerti Shekhawat

- Attention diagnostics
- Visual evidence and attention/region analysis
- Reliability diagnostics
- Visualization and qualitative analysis
- Attention–saliency analysis
- Demonstration and evidence pipeline

### Shared Contributions 

- Research methodology and experimental design
- Literature review and research analysis
- Definition and refinement of research questions
- Experimental result interpretation
- Major project report and presentation preparation
- Validation of findings and final project conclusions
  
---

## 📄 Research Documentation

Detailed methodology and supporting documentation are maintained under:

```text
docs/
```

The attention diagnostic methodology is documented in:

```text
docs/attention_pipeline.md
```

---

## 📚 Citation

If referencing this work, please use the citation information provided in:

```text
CITATION.cff
```

---

## 📄 License

MIT License
