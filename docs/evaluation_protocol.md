# Evaluation Protocol

## 1. Purpose

This document defines the reproducible evaluation protocol for the Major project.

The evaluation is designed to answer two related questions:

1. How do the evaluated image-captioning models perform under conventional automatic caption metrics?
2. How well do those automatic metrics reflect human-judged semantic correctness?

The protocol therefore evaluates model outputs using both automatic metrics and human semantic annotations, and explicitly analyzes cases where the two disagree.

The implementation is split between reusable code under `src/evaluation/` and the orchestration notebook:

```text
notebooks/major_evaluation.ipynb
        │
        ▼
src/evaluation/
        │
        ├── data_validation.py
        ├── metrics.py
        ├── human_alignment.py
        ├── error_analysis.py
        ├── comparison.py
        └── reporting.py
        │
        ▼
results/evaluation/
```

---

## 2. Evaluation Inputs

### 2.1 Generated captions

Primary input:

```text
data/annotations/captions_all_models (1).csv
```

The generated-caption file must contain:

- a stable image identifier;
- ground-truth/reference captions;
- one generated-caption column for every evaluated model.

The current model-to-column mapping is:

| Model | Caption column |
|---|---|
| BLIP | `blip_caption` |
| BLIP-2 | `blip2_caption` |
| OFA | `ofa_caption` |
| ViT-GPT2 | `vit_gpt2_caption` |

The validation stage must fail loudly if an expected model column is absent rather than silently skipping that model.

### 2.2 Human annotations

Primary input:

```text
data/annotation/manual_labels.csv
```

Human annotations are treated as experimental data.

The Major evaluation must not silently replace missing real annotations with simulated labels.

The annotation workflow must validate:

- image identifiers;
- model identifiers;
- annotation coverage;
- duplicate image-model annotations;
- correctness labels;
- semantic error categories;
- consistency between correctness and error type.

---

## 3. Dataset and Split Policy

The evaluation must use the dataset and split represented by the generated-caption input file.

The exact dataset size, image subset, and train/validation/test split should be reported from the actual input metadata and not manually invented in this document.

All models must be evaluated on the same image set wherever the generated-caption file provides complete coverage.

If a model has missing predictions for an image, that missingness must be reported and investigated rather than silently dropping the image for only that model.

---

## 4. Data Integrity Checks

Before metric computation, the evaluation pipeline must validate the input data.

### Required checks

1. Required files exist.
2. Required columns exist.
3. Image identifiers are non-null.
4. Image identifiers are unique where uniqueness is required.
5. Model caption columns contain valid values.
6. Ground-truth references are present and parseable.
7. Reference-caption lists have the expected structure.
8. Duplicate image-model records are detected.
9. Missing model predictions are detected.
10. Human annotation coverage is checked against the evaluated models/images.
11. Annotation categories are normalized to the canonical taxonomy.
12. Invalid or contradictory human labels cause an explicit validation failure.

The pipeline must distinguish between:

- a genuinely missing observation;
- an invalid observation;
- a valid observation with a low metric score.

These cases must never be silently converted into the same value.

---

## 5. Automatic Metrics

The automatic evaluation uses:

- BLEU;
- CIDEr;
- METEOR;
- ROUGE-L.

The implementation lives in:

```text
src/evaluation/metrics.py
```

### 5.1 BLEU

BLEU is computed using the available ground-truth references and generated caption.

The current metric implementation records sentence/image-level BLEU-1 through BLEU-4 values and uses smoothing where required for sentence-level evaluation.

The Major report must describe these values as sentence/image-level evaluation rather than presenting them as corpus-level BLEU.

### 5.2 CIDEr

CIDEr measures similarity between the generated caption and reference captions using TF-IDF-weighted n-gram representations.

The project implementation uses a self-contained TF-IDF n-gram cosine formulation with the project-defined scoring procedure.

The report must distinguish this implementation from claiming exact reproduction of an external official COCO evaluation package unless that official evaluator is actually used.

### 5.3 METEOR

METEOR is computed against the available reference captions.

For multi-reference evaluation, the implementation uses the strongest reference-level score for the generated caption.

### 5.4 ROUGE-L

ROUGE-L is based on longest-common-subsequence overlap.

For multiple references, the evaluation uses the strongest reference-level score.

---

## 6. Metric Aggregation

Metric values are first retained at the image/caption level.

Model-level summaries should report appropriate descriptive statistics, including:

- mean;
- standard deviation;
- number of evaluated examples.

The aggregation procedure must be identical across models.

Metric failures must not be silently converted to zero.

Invalid metric computations should be surfaced as errors or explicit missing values according to the implementation contract.

---

## 7. Human Semantic Evaluation

Human evaluation is used to determine whether generated captions are semantically correct.

The canonical taxonomy is:

| Code / Class | Meaning |
|---|---|
| `correct` | Caption is semantically correct |
| `attribute_mismatch` | An object/property attribute is incorrect |
| `relation_mismatch` | A relation, action, spatial relationship, or interaction is incorrect |
| `object_misidentification` | A visible object/entity is identified as the wrong object/entity |
| `object_hallucination` | The caption introduces an object/entity not supported by the image |

The implementation must preserve these categories separately.

In particular, `object_misidentification` and `object_hallucination` must not be collapsed into one generic object-error category for the final Major analysis.

---

## 8. Annotation Normalization

Human annotation data may arrive in a supported wide or long representation, but it must be normalized before analysis.

The normalized representation should preserve, at minimum:

```text
image_id
model
human_correct
error_type
```

If the annotation source contains additional fields such as:

```text
primary_error_token
notes
review_status
```

they should be preserved when useful for qualitative analysis.

### Consistency rules

- `correct` must correspond to human correctness.
- A non-correct semantic error must have one canonical error type.
- Unsupported error labels must be rejected.
- Duplicate image-model annotations must be resolved before final analysis.
- Missing annotations must be reported.

---

## 9. Inter-Annotator Agreement

Where multiple human annotators are available, agreement must be measured from the real annotations.

The evaluation should report:

- pairwise Cohen's kappa where applicable;
- Krippendorff's alpha for the multi-annotator nominal annotation setting where applicable.

Missing annotations must not automatically be interpreted as disagreement.

Agreement statistics must be interpreted as evidence about annotation consistency, not as evidence that the automatic metric is correct.

---

## 10. Metric–Human Alignment

The central analysis compares automatic metric judgments with human semantic correctness.

For the exploratory binary analysis, BLEU-4 uses:

```text
threshold = 0.20
```

This threshold is an exploratory decision boundary.

It must **not** be described as:

- a probability;
- a calibrated confidence score;
- a universally valid semantic-correctness threshold.

The binary comparison can identify:

- true high / human correct;
- false high / metric high but human incorrect;
- true low / metric low and human incorrect;
- false low / metric low but human correct.

The false-high and false-low cases are particularly important because they expose metric–human misalignment.

---

## 11. Model Comparison

Models are compared jointly across:

### Automatic performance

- BLEU-1;
- BLEU-2;
- BLEU-3;
- BLEU-4;
- CIDEr;
- METEOR;
- ROUGE-L.

### Human semantic performance

- human correctness rate;
- semantic error distribution;
- annotation agreement where available.

### Metric reliability / misalignment

- false-high rate;
- false-low rate;
- metric-human agreement;
- qualitative examples of disagreement.

A model must not be declared universally superior from one automatic metric alone.

---

## 12. Semantic Error Analysis

The evaluation should quantify the distribution of:

```text
attribute_mismatch
relation_mismatch
object_misidentification
object_hallucination
```

for each evaluated model.

The analysis should also preserve representative qualitative examples.

Examples should contain enough context to understand:

- the generated caption;
- the human semantic judgment;
- the error category;
- the relevant metric value;
- why the metric and human judgment agree or disagree.

This qualitative analysis is necessary to support claims about *why* conventional metrics fail, rather than only showing aggregate numbers.

---

## 13. Reproducible Output Structure

Evaluation outputs belong under:

```text
results/evaluation/
```

Expected artifacts include, where produced by the implementation:

```text
results/evaluation/
├── metrics_all_models.csv
├── metric_summary.csv
├── normalized_human_annotations.csv
├── iaa_scores.csv
├── error_type_distribution.csv
├── misalignment_summary.csv
├── metric_classifier_report.csv
├── model_comparison.csv
└── qualitative_error_examples.csv
```

Additional plots or tables may be generated by the reporting layer.

Generated results should be derived from one validated evaluation run and should not be manually edited.

---

## 14. Reproducibility Requirements

The evaluation must be reproducible from a clean repository checkout.

The following should be fixed or explicitly recorded:

- model identifiers;
- input file paths;
- metric implementations;
- preprocessing/tokenization behavior;
- reference-selection behavior;
- aggregation rules;
- human-annotation taxonomy;
- BLEU-4 exploratory threshold;
- software dependencies;
- random seeds where randomness exists.

The reusable evaluation implementation belongs in `src/evaluation/`.

The notebook:

```text
notebooks/major_evaluation.ipynb
```

should act primarily as an orchestration and inspection layer.

---

## 15. Recommended Execution Order

Run the Major evaluation in this order:

```text
1. Validate generated captions
        ↓
2. Validate human annotations
        ↓
3. Normalize annotation schema
        ↓
4. Compute automatic metrics
        ↓
5. Compute model-level metric summaries
        ↓
6. Compute inter-annotator agreement
        ↓
7. Compute semantic error distributions
        ↓
8. Compute metric–human alignment
        ↓
9. Extract false-high / false-low examples
        ↓
10. Compare models
        ↓
11. Generate reproducible tables
        ↓
12. Run final integrity checks
```

The evaluation should stop if a critical validation step fails.

---

## 16. Interpretation Rules

The Major evaluation must avoid the following unsupported conclusions:

### Do not claim

> A high BLEU/CIDEr/METEOR/ROUGE-L score proves semantic correctness.

Instead, evaluate the empirical relationship between the metric and human judgments.

### Do not claim

> BLEU-4 ≥ 0.20 means the caption is semantically correct.

The threshold is exploratory.

### Do not claim

> The best automatic metric is automatically the best semantic metric.

This requires comparison against human annotations.

### Do not hide

- missing predictions;
- invalid references;
- annotation disagreements;
- metric failures;
- taxonomy ambiguity;
- model-specific coverage differences.

These are part of the evaluation record.

---

## 17. Major Research Question

The evaluation protocol is ultimately intended to support the Major project's central empirical analysis:

> **To what extent do conventional image-captioning metrics agree with human semantic correctness, and what types of semantic errors reveal systematic metric–human misalignment?**

The answer should be supported through:

1. automatic metric results;
2. human correctness judgments;
3. inter-annotator agreement;
4. semantic error taxonomy;
5. metric–human confusion/misalignment analysis;
6. model-level comparisons;
7. qualitative false-high and false-low examples.

No single metric should be treated as the sole definition of caption quality.

---

## 18. Source of Truth

The executable implementation is the source of truth for numerical results.

This document defines the evaluation contract and interpretation rules.

The following should remain synchronized:

```text
docs/evaluation_protocol.md
        ↕
src/evaluation/
        ↕
notebooks/major_evaluation.ipynb
        ↕
results/evaluation/
```

If an implementation detail changes, update this protocol and the relevant code/documentation together.
