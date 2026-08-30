# Model Card — AWARE (Large v4)

## Overview

Sentence-level, multi-label classifier for **Community Cultural Wealth (CCW)** themes in
student reflective essays. DeBERTa-v3-large (438M) with domain-adaptive pre-training, a BiLSTM
essay-context encoder, asymmetric loss, and per-theme threshold optimization.

- **Developed by:** Khalid Mehtab Khan, San Francisco State University (M.S. thesis, 2026)
- **Domain:** student reflective writing in undergraduate STEM courses (ALMA Project, SFSU)
- **Labels:** 8 CCW themes, multi-label — Navigational, Aspirational, Perseverance, Social,
  Resistance, Familial_Capital, Spiritual, Attainment (plus `class_0` for no theme)

## Intended use

**In scope**
- Accelerating research annotation: pre-annotating uncoded essays so human coders review rather
  than start from scratch
- Aggregate research analysis of how CCW themes appear across courses, semesters, and prompts
- Screening for **single-theme** content, where it is strongest

**Explicitly out of scope**
- **Any evaluation, grading, scoring, or ranking of individual students.** The model classifies
  themes present in writing, not the quality of the writing or the worth of the writer.
- Identifying students who do or do not express particular forms of cultural capital
- High-stakes or individual decisions of any kind
- Domains outside student reflective writing, where it has not been validated

## Performance

Test set: 273 essays / 1,838 sentences. **Macro-F1 0.494** [0.464, 0.520], PR-AUC 0.484,
ROC-AUC 0.888. That is **+30.8%** over a TF-IDF + SVM baseline.

| Theme | Prevalence | F1 |
|---|---|---|
| Navigational | 24.0% | 0.707 |
| Familial_Capital | 3.9% | 0.600 |
| Aspirational | 15.4% | 0.585 |
| Social | 5.6% | 0.506 |
| Perseverance | 8.3% | 0.480 |
| Resistance | 4.5% | 0.395 |
| Spiritual | 3.4% | 0.375 |
| Attainment | 2.4% | 0.303 |

**Read this alongside the headline number.** On **single-theme sentences**, F1 is
**0.568–0.896 with perfect precision across all eight themes**. The aggregate figure is dragged
down by **multi-label disentanglement** — separating themes that co-occur in one sentence — not
by an inability to detect themes. Practically: deployment-ready for single-theme screening,
not yet for multi-label analysis.

## Limitations

- **Wrong roughly half the time in aggregate.** At F1 = 0.494, any individual prediction is a
  hypothesis, not a fact.
- **Rare themes are weak.** Attainment (0.303) and Spiritual (0.375) suffer from scarcity and
  low separability; three of eight themes have d-prime ≤ 1.0.
- **Label noise from excerpt-to-sentence propagation.** Annotators coded excerpts; those labels
  were propagated to constituent sentences, so some sentences carry labels they do not
  individually express.
- **No inter-annotator agreement data**, so annotation reliability could not be quantified.
- **Overfitting persists** despite the regularization stack (train–test F1 gap 0.265).
- **Single train/validation/test split**; no cross-validation.
- **Incomplete annotation coverage** — early batches predate parts of the coding protocol.

## Bias

The model learns from human annotations and will reproduce and amplify any systematic bias in
them. If annotators were more likely to identify a theme in essays from particular demographic
groups, the model inherits that. **Without inter-annotator agreement data, the magnitude of this
risk is unmeasured.** Bias auditing across demographic subgroups is required future work and has
not been performed.

## Ethical constraints

CCW themes are sociocultural constructs requiring interpretive judgment. Automating their
detection risks flattening that judgment.

Predictions on student essays carry **the same confidentiality obligations as the essays
themselves**. Outputs are for aggregate research analysis and annotation acceleration only,
never individual student evaluation. Deployments should present predictions alongside human
annotations with confidence scores and a visible source distinction, as the dashboard does.

## Training data

17,622 sentences from 2,636 annotated ALMA essays; imbalance up to 21.8:1. The corpus is
**not publicly distributed** — see [`../data/README.md`](../data/README.md).
