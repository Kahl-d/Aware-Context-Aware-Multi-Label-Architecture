# AWARE

### Automated classification of Community Cultural Wealth themes in student reflective writing

AWARE is a sentence-level, multi-label classification framework for identifying **Community
Cultural Wealth (CCW)** themes in student reflective essays. It was built for the
[ALMA Project](https://alma.sfsu.edu) at San Francisco State University, where six trained
coders annotated 2,710 essays by hand and 1,388 essays remain uncoded because manual coding
cannot keep pace with collection.

This repository accompanies the M.S. thesis
**[AWARE: A Framework for Automated Classification of Community Cultural Wealth Themes](thesis/AWARE-thesis-Khan-2026.pdf)**
(Khalid Mehtab Khan, Data Science & Artificial Intelligence, SFSU, May 2026).

> **AWARE recognizes student strengths. It does not evaluate, score, or rank students.**
> Model output is a research aid and a hypothesis, never a verdict about a person.

---

## Why the problem is hard

CCW themes are sociocultural constructs that overlap heavily in meaning, so a sentence can
carry zero, one, or several themes at once. Before training anything, the data analysis
quantified the difficulty:

| Property | Measurement |
|---|---|
| Nearest-neighbour confusability | **38.7%** leave-one-out KNN error across themes |
| Theme separability | **3 of 8** themes have d-prime **≤ 1.0** |
| Class imbalance | up to **21.8 : 1** |
| Final dataset | **17,622 sentences** from **2,636 essays** |

These numbers predicted per-theme performance before a single model was trained, which is the
first practical lesson of the thesis: characterize the data first.

## What AWARE does about it

Four components, each targeting a specific failure mode:

1. **Domain-adaptive pre-training (DAPT)** on the essay corpus, closing the gap between
   general-web pretraining and student reflective writing.
2. **Essay-level context encoding** — a BiLSTM over sentence embeddings, so each sentence is
   read in the context of the narrative around it rather than in isolation.
3. **Asymmetric loss** with class-balanced weighting, for severe imbalance.
4. **Per-theme threshold optimization**, which recovers rare-class recall and contributed
   **+0.084 Macro-F1** on its own, more than any architectural choice.

Backbone: **DeBERTa-v3**, evaluated at base (86M) and large (438M) scale.

## Results

Evaluated on a held-out test set of 273 essays / 1,838 sentences.

| Model | Test Macro-F1 | Test PR-AUC |
|---|---|---|
| Majority class | 0.000 | 0.085 |
| Random (prior) | 0.084 | 0.089 |
| TF-IDF + LogReg | 0.187 | 0.363 |
| TF-IDF + SVM | 0.274 | 0.333 |
| Base Standard | 0.487 | **0.507** |
| Large Standard | 0.483 | 0.496 |
| Base AWARE | 0.472 | 0.480 |
| **Large AWARE (v4)** | **0.494** | 0.484 |

**Macro-F1 0.494** [0.464, 0.520], a **30.8% improvement over the TF-IDF baseline**.

Two results are reported honestly rather than selectively. First, Large AWARE wins on F1 while
**Base Standard wins on PR-AUC (0.507)** — the ranking inverts depending on the metric, because
F1 depends on the tuned decision threshold and PR-AUC does not. Second, and more importantly:

> Aggregate Macro-F1 **understates** what the model can do. On **single-theme sentences**, F1
> ranges **0.568–0.896 with perfect precision across all eight themes.** The bottleneck is not
> theme detection. It is **multi-label disentanglement** — separating themes that co-occur in
> the same sentence.

That distinction changes the recommendation: the model is ready for single-theme screening and
needs refinement for multi-label analysis.

### Per-theme F1 (Large AWARE v4)

| Theme | Prevalence | F1 |
|---|---|---|
| Navigational | 24.0% | 0.707 |
| Aspirational | 15.4% | 0.585 |
| Social | 5.6% | 0.506 |
| Perseverance | 8.3% | 0.480 |
| Familial_Capital | 3.9% | 0.600 |
| Resistance | 4.5% | 0.395 |
| Spiritual | 3.4% | 0.375 |
| Attainment | 2.4% | 0.303 |

Performance tracks prevalence and separability, with one instructive exception:
**Familial_Capital** is rare (3.9%) yet scores 0.600, because it is lexically distinctive.
Rarity alone does not determine difficulty; semantic distinctness matters more.

---

## Repository map

| Directory | What is in it |
|---|---|
| [`thesis/`](thesis/) | The full thesis PDF, the working paper, the defense presentation, and conference posters. |
| [`docs/`](docs/) | How the system works: data pipeline, model inference, key results. Start here. |
| [`data/`](data/) | Schemas, corpus statistics, and a pseudonymized sample. **The corpus itself is not distributed — see [`data/README.md`](data/README.md).** |
| [`pipeline/`](pipeline/) | The data processing pipeline: extraction, semantic cleaning, dataset construction (V1→V4). |
| [`models/`](models/) | The AWARE architecture, training configuration, and the [model card](models/MODEL_CARD.md). |
| [`experiments/`](experiments/) | The 2×2 factorial (scale × approach) plus baselines: configs, SLURM jobs, logs, results. |
| [`dashboard/`](dashboard/) | The ALMA Research Dashboard — corpus explorer and annotation workbench. |
| [`archive/`](archive/) | AWARE v2, the earlier 11-theme line, kept for provenance. |

## Ethics

Student reflective writing is personal. The essays discuss family, identity, faith, community,
and adversity. Even with numeric IDs replacing names, essay content can be re-identified by
anyone holding a course roster, so **the corpus is not published here** and model predictions
carry the same confidentiality obligations as the essays themselves. Section 7.6 of the thesis
states this in full; [`data/README.md`](data/README.md) explains what is available and how to
request access through the ALMA team.

The model must never be used to evaluate, grade, or rank individual students.

## Citation

See [`CITATION.cff`](CITATION.cff).

```bibtex
@mastersthesis{khan2026aware,
  title  = {AWARE: A Framework for Automated Classification of Community Cultural Wealth Themes},
  author = {Khan, Khalid Mehtab},
  school = {San Francisco State University},
  year   = {2026},
  type   = {M.S. thesis},
  note   = {Data Science and Artificial Intelligence}
}
```

## Acknowledgements

Thesis committee: **Dr. Anagha Kulkarni** (chair), **Dr. Kim Coble**, **Dr. Anisha Singh**.
This work rests on the ALMA Project team at SFSU, and above all on the annotators whose careful
coding of thousands of essays over multiple semesters created the dataset behind it.
