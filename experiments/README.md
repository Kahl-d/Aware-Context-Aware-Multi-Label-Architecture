# Experiments

A **2×2 factorial** isolating two independent variables — model **scale** (base vs large) and
**approach** (standard fine-tuning vs the AWARE framework) — plus a baseline suite that
establishes the difficulty floor.

| Directory | Scale | Approach |
|---|---|---|
| [`1_base_standard/`](1_base_standard/) | DeBERTa-v3-base (86M) | Standard fine-tuning |
| [`2_base_aware/`](2_base_aware/) | DeBERTa-v3-base (86M) | AWARE |
| [`3_large_standard/`](3_large_standard/) | DeBERTa-v3-large (438M) | Standard fine-tuning |
| [`4_large_aware/`](4_large_aware/) | DeBERTa-v3-large (438M) | **AWARE — the reported model** |
| [`5_baselines/`](5_baselines/) | — | Majority class, random prior, TF-IDF + LogReg, TF-IDF + SVM |

## Results

| Model | Test Macro-F1 | Test PR-AUC | ROC-AUC |
|---|---|---|---|
| Majority class | 0.000 | 0.085 | — |
| Random (prior) | 0.084 | 0.089 | — |
| TF-IDF + LogReg | 0.187 | 0.363 | — |
| TF-IDF + SVM | 0.274 | 0.333 | — |
| Base Standard | 0.487 | **0.507** | 0.895 |
| Large Standard | 0.483 | 0.496 | 0.894 |
| Base AWARE | 0.472 | 0.480 | 0.895 |
| **Large AWARE (v4)** | **0.494** | 0.484 | 0.888 |

**The interaction is the finding.** AWARE *hurts* at base scale (0.472 vs 0.487) and *helps* at
large scale (0.494 vs 0.483). The framework's regularization is overhead a small model does not
need and a large model does. Reporting only the winning cell would have hidden this.

**F1 and PR-AUC disagree.** Large AWARE leads on F1; Base Standard leads on PR-AUC. F1 depends
on the tuned decision threshold, PR-AUC does not, so the two answer different questions. Both
are reported. See §6.4.3 of the thesis.

## The v3 failure

`4_large_aware` is version **4**. Version 3 — the same large model with base-scale
hyperparameters — performed *worse than base*. Seven distinct failure modes were diagnosed and
fixed. The lesson generalizes: **scaling requires re-tuning, not just re-running.** Optimization
dynamics change with model scale, and the configuration must change with them. Chapter 5.4–5.5
documents the failure and the recovery.

## Layout of a run

```
<run>/
├── configs/      YAML training configuration
├── scripts/      training, DAPT, dataset construction, evaluation, losses, metrics
├── logs/         SLURM stdout/stderr from the actual runs
├── results/      metrics, per-theme breakdowns, DAPT encoder artifacts
├── job_1_dapt.sh    SLURM: domain-adaptive pre-training
└── job_2_train.sh   SLURM: fine-tuning
```

Training data is **not** included — see [`../data/README.md`](../data/README.md). Logs and
configs are kept verbatim so the runs are auditable.

## Analysis documents

- [`RESULTS_COMPREHENSIVE.md`](RESULTS_COMPREHENSIVE.md) — full comparison, all models, all themes
- [`OBSERVATIONS_AND_FINDINGS.md`](OBSERVATIONS_AND_FINDINGS.md) — what was learned
- [`PAPER_OBSERVATIONS.md`](PAPER_OBSERVATIONS.md) — findings framed for publication
- [`RUN_CHECKLIST.md`](RUN_CHECKLIST.md) — the execution protocol
