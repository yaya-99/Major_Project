# Attention Diagnostic Pipeline

This document describes the attention-analysis pipeline used in the Major Project.

The purpose of this pipeline is **diagnostic**: to examine whether model attention is directed toward visually relevant regions and to provide supporting evidence about the relationship between perceptual focus and semantic correctness.

Attention alignment is **not** used as the Major's training objective. The proposed training intervention instead operates on contextual token representations and visual-semantic representations through a targeted consistency objective.

---

## 1. Role in the Major Research Pipeline

The Major separates three related but distinct components:

1. **Training pipeline** — learns a targeted semantic-grounding consistency objective.
2. **Diagnostic pipeline** — analyzes token-level cross-attention and its spatial alignment.
3. **Evaluation pipeline** — measures semantic correctness and hallucination independently.

This separation is important because attention alignment and semantic correctness are not equivalent.

The diagnostic attention pipeline therefore provides evidence about **where the model is looking**, while the evaluation pipeline determines **whether the generated content is semantically correct**.

---

## 2. Relationship to the Minor

The Minor Project established the motivation for examining attention behavior in image captioning models.

The Minor analyzed pretrained captioning models including BLIP, BLIP-2, and ViT-GPT2 by extracting token-level decoder cross-attention, aggregating the attention across heads, resizing the resulting maps to a common spatial resolution, and comparing them with human saliency information.

The Major retains this attention analysis as supporting evidence, but does not extend the Minor's multi-model attention study into the Major training objective.

Instead, the Major focuses on a single full VLM backbone and introduces a targeted consistency objective intended to reduce semantic grounding failures.

---

## 3. Diagnostic Attention Pipeline

The diagnostic pipeline is:

```text
COCO image
    ↓
Pretrained / trained BLIP model
    ↓
Generated caption
    ↓
Token-level decoder cross-attention
    ↓
Attention extraction
    ↓
Head/layer aggregation
    ↓
Spatial normalization
    ↓
Token-level spatial attention map
    ↓
Optional comparison with human saliency
    ↓
Attention–saliency divergence / visualization
```

The resulting token-level attention distributions are used for mechanism analysis and visualization.

---

## 4. Caption Generation

Images are passed through the captioning model using the normal inference/generation procedure.

For each generated caption:

1. The generated token sequence is retained.
2. Decoder cross-attention information is requested when supported by the model.
3. Attention associated with each generated token is extracted.
4. The resulting attention distribution is associated with the corresponding generated token.

This diagnostic procedure operates on **generated captions**.

It should not be confused with the Major training representation, which is extracted under **teacher forcing using the ground-truth caption**.

---

## 5. Cross-Attention Extraction

For each generated token:

1. Retrieve the decoder cross-attention tensor.
2. Select the attention values corresponding to image/visual feature tokens.
3. Aggregate across attention heads.
4. Aggregate across layers when the diagnostic configuration requires it.
5. Preserve the resulting token-level spatial distribution.

Conceptually:

```text
cross-attention
    [batch, heads, text positions, visual positions]
                    ↓
              select token
                    ↓
             aggregate heads
                    ↓
             aggregate layers
                    ↓
        token-level spatial map
```

The exact tensor dimensions and native visual-token count are determined from the architecture probe for the Major's locked BLIP checkpoint rather than assumed from the Minor implementation.

---

## 6. Spatial Representation

Attention maps produced by transformer models may be represented over visual feature tokens rather than directly as image-resolution maps.

For diagnostic comparison, the extracted attention maps are converted to a common spatial representation.

The Minor analysis used a **24 × 24 standardized analysis grid**. In the Major documentation, this should be interpreted as an analysis-resolution convention rather than as an assumption about the model's native visual-token grid.

Where necessary:

1. Convert the extracted visual-token attention to a spatial arrangement.
2. Resize to the standardized diagnostic resolution.
3. Normalize the spatial values so that they form a probability distribution.

The spatial representation is therefore suitable for token-level visualization and comparison.

---

## 7. Human Saliency Comparison

Human visual attention may be approximated using SALICON saliency maps.

For diagnostic comparison:

1. Load the corresponding human saliency map.
2. Normalize the saliency values.
3. Resize the saliency map to the same diagnostic spatial resolution.
4. Convert it into a probability distribution.

The comparison is:

```text
Model token attention        Human visual saliency
        ↓                              ↓
 spatial normalization         spatial normalization
        ↓                              ↓
 probability distribution     probability distribution
        └──────────────┬───────────────┘
                       ↓
              divergence analysis
```

SALICON is used only as a **diagnostic human-attention reference**.

It is not used as the visual-semantic target for the Major's consistency loss.

---

## 8. Attention–Saliency Alignment

For each token, the normalized model-attention distribution can be compared with the normalized human-saliency distribution.

The diagnostic analysis may use:

- Jensen–Shannon (JS) divergence
- Kullback–Leibler (KL) divergence
- spatial attention visualizations

These quantities describe the similarity or divergence between model perceptual focus and human saliency.

They do **not** directly measure whether the generated token is semantically correct.

---

## 9. Relationship to Semantic Correctness

The Major explicitly separates:

```text
Perceptual focus
    ↓
cross-attention analysis

Semantic correctness
    ↓
hallucination / grounding evaluation

Representation-level grounding
    ↓
consistency objective
```

Therefore:

> Strong attention alignment does not by itself establish semantic correctness.

A model can attend to a visually relevant region while still producing an incorrect semantic description.

This is precisely why the Major evaluates semantic correctness independently and introduces a representation-level consistency objective.

---

## 10. Relationship to the Major Consistency Objective

The attention diagnostic pipeline should not be confused with the proposed training mechanism.

### Diagnostic pipeline

```text
Image
  ↓
BLIP generation
  ↓
Generated token
  ↓
Cross-attention
  ↓
Spatial attention map
  ↓
Attention analysis
```

### Major training pipeline

```text
Image ───────────────→ Visual encoder
                           ↓
                  visual-semantic representation
                           ↓
                    frozen projection
                           ↓
                           f_v

Ground-truth caption
        ↓
BLIP decoder under teacher forcing
        ↓
contextual token hidden state
        ↓
trainable token projection
        ↓
        f_s

             f_s ↔ f_v
                 ↓
       targeted consistency loss
                 ↓
              InfoNCE
```

The training objective operates on representation similarity, not directly on cross-attention maps.

---

## 11. Token Eligibility and Diagnostic Scope

The Major's consistency objective is applied only to eligible caption tokens.

Eligibility is determined using the project's token/region selection procedure, including:

- concrete noun part-of-speech categories,
- detector confidence,
- regional uniqueness.

Tokens that are not eligible for the consistency objective still participate in the normal captioning loss.

The attention diagnostic may be used to visualize generated tokens more broadly, but its interpretation should remain separate from the training eligibility mechanism.

---

## 12. Training, Diagnostic, and Evaluation Separation

The complete Major research flow is:

```text
                         ┌──────────────────────────────┐
                         │       Training Pipeline       │
                         │                              │
Image + GT caption ────→ │ contextual token features   │
                         │ + visual-semantic features  │
                         │ + targeted consistency loss │
                         └──────────────┬───────────────┘
                                        │
                                        ↓
                                trained model
                                        │
                         ┌──────────────┼──────────────┐
                         ↓              ↓              ↓
                  Generation       Attention       Evaluation
                         │           Diagnostic          │
                         ↓              │                ↓
                    captions     cross-attention    semantic metrics
                                        │                │
                                        ↓                ↓
                                  visual evidence   correctness evidence
```

This structure prevents the study from conflating:

- where the model attends,
- what representation it learns,
- and whether its generated statement is correct.

---

## 13. Outputs

The diagnostic pipeline can produce:

- token-level normalized attention maps,
- token-level spatial distributions,
- human saliency distributions,
- JS divergence values,
- KL divergence values,
- attention visualizations,
- token-level diagnostic tables.

These outputs should be stored separately from the main training checkpoints and prediction/evaluation artifacts.

Suggested locations:

```text
results/
├── attention/
│   ├── maps/
│   ├── distributions/
│   └── diagnostics/
├── tables/
└── figures/
```

The exact directory structure should follow the Major repository configuration.

---

## 14. Interpretation of Diagnostic Results

Attention results should be interpreted as supporting mechanism evidence.

For example:

- attention concentrated on a relevant object region may indicate appropriate perceptual focus;
- attention aligned with human saliency may indicate human-like perceptual focus;
- low attention–saliency divergence does not prove semantic correctness;
- a semantically incorrect caption despite relevant attention is evidence of the type of failure motivating the Major intervention.

The diagnostic analysis should therefore be interpreted jointly with semantic-grounding evaluation.

---

## 15. Implementation

The Major implementation should use the locked BLIP checkpoint and the architecture information obtained from the BLIP architecture probe.

The implementation should not hard-code architectural assumptions from the Minor analysis.

In particular, the implementation must verify:

- decoder hidden-state dimensionality,
- visual-token count,
- cross-attention tensor shape,
- token-position alignment,
- supported attention outputs.

The architecture probe in:

```text
notebooks/02_blip_architecture_and_generation.ipynb
```

is the authoritative source for these implementation details.

---

## 16. Important Distinction

The Major contains three different kinds of evidence:

| Evidence | What it answers |
|---|---|
| Cross-attention diagnostics | **Where is the model looking?** |
| Representation consistency | **Are token and visual-semantic representations aligned?** |
| Semantic-grounding evaluation | **Is the generated statement correct?** |

The central Major contribution is the **targeted consistency objective**.

Attention analysis is retained as diagnostic and mechanism evidence rather than being treated as the training objective itself.
