# Thesis, paper, and presentations

| Item | File |
|---|---|
| **M.S. thesis (full)** | [`AWARE-thesis-Khan-2026.pdf`](AWARE-thesis-Khan-2026.pdf) |
| Working paper | [`paper/AWARE_Beyond_Sentence_Boundaries.pdf`](paper/) |
| Defense presentation | [`presentation/`](presentation/) |
| Conference posters and abstracts | [`posters/`](posters/) — PERC 2026 |

**AWARE: A Framework for Automated Classification of Community Cultural Wealth Themes**
Khalid Mehtab Khan · M.S. Data Science & Artificial Intelligence
San Francisco State University · May 2026

Committee: Dr. Anagha Kulkarni (chair) · Dr. Kim Coble · Dr. Anisha Singh

## Chapters

1. Introduction — motivation, problem analysis, contributions
2. Related Work — CCW, multi-label classification, DeBERTa, DAPT, imbalance, calibration
3. Data — the ALMA corpus, the four-stage pipeline, dataset analysis, splits
4. Methodology — representation, DAPT, the AWARE architecture, losses, training, evaluation
5. Experiments — baselines through the 2×2 factorial, including the v3 failure
6. Results — RQ1/2/3, per-theme analysis, the multi-label bottleneck, calibration
7. Discussion — interpretation, limitations, the dashboard, ethics, implications
8. Conclusion — answers, contributions, future work

## Research questions

**RQ1** — Can a domain-adapted transformer effectively classify multiple CCW themes
simultaneously at the sentence level?
**RQ2** — How does model scale affect performance, and what engineering challenges arise when
scaling on small, imbalanced datasets?
**RQ3** — What are the fundamental challenges, and which themes are most and least amenable to
automated detection?
