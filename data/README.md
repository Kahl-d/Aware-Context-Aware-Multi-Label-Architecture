# Data

## What is not here, and why

**The ALMA essay corpus is not distributed in this repository.**

The essays are student reflective writing about family, identity, community, faith, and
adversity. Section 7.6 of the thesis states the constraint directly:

> "While the data is anonymized (numeric IDs replace student names), the essay content could
> potentially be re-identified by someone with access to the course roster. Model predictions
> on these essays must be treated with the same confidentiality protections as the essays
> themselves."

Anonymization by ID substitution does not make reflective narrative safe to publish. A student
describing their specific circumstances, in a named course and semester, is identifiable to
anyone with the roster. That risk does not expire, and it is not the author's alone to accept:
the corpus belongs to the ALMA Project and its participants.

**Requesting access:** contact the ALMA Project research team at San Francisco State University.
Access is governed by the project's IRB protocol.

## What is here

| Path | Contents |
|---|---|
| `samples/` | A small, pseudonymized sample: 2 essays and their 18 sentence-level records, with labels, tags, and predictions. Enough to see the exact shape of the data. |
| `samples/dataset_versions.json` | Version metadata for V1→V4 (sentence, essay, and theme counts). No student text. |
| `stats/splits_stats.json` | Train / validation / test split statistics. |

Corpus-level statistics — class distribution, co-occurrence, separability, annotation density —
are reported in [`../docs/`](../docs/) and Chapter 3 of the thesis, and are served by the
[corpus explorer](../dashboard/) without exposing essay text.

## The dataset, described

Built by the pipeline in [`../pipeline/`](../pipeline/) from ALMA source files.

| Property | Value |
|---|---|
| Essays annotated | 2,710 by six trained coders; **2,636** after processing |
| Essays unannotated | 1,388 (the motivation for automation) |
| Final sentences (V4) | **17,622** |
| Removed in semantic cleaning | 1,705 sentences (8.6%) |
| Themes | 8, consolidated from an original 11 |
| Class imbalance | up to 21.8 : 1 |
| Split | 2,095 train essays (14,023 sentences) / 273 test essays (1,838 sentences) |

### The eight themes

`Navigational` · `Aspirational` · `Perseverance` · `Social` · `Resistance` ·
`Familial_Capital` · `Spiritual` · `Attainment`

Plus `class_0` for sentences carrying no CCW theme. Labels are multi-label: a sentence may
carry zero, one, or several themes at once.

### Dataset versions

| Version | What changed |
|---|---|
| V1 | Original processed extraction |
| V2 | Merged and cleaned |
| V3 | Sentence-boundary cleaned |
| V4 | Theme consolidation (11 → 8); **the dataset all reported results use** |

## Reproducing the data locally

With ALMA source access, run the pipeline in order — see
[`../pipeline/README.md`](../pipeline/README.md). The dashboard consumes the same V4 dataset
the model trains on, so there is no version drift between what a researcher browses and what
the model learned.
