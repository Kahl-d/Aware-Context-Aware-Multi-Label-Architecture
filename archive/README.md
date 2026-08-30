# Archive — AWARE v2

The earlier research line, kept for provenance. **This is not the thesis result.**

AWARE v2 worked over **11 CCW themes** and a much larger, noisier extraction — 86,054 sentences
from 12,144 essays, split by student ID with zero leakage, at 72× imbalance. It was superseded
when theme consolidation (11 → 8) and embedding-based semantic cleaning produced the smaller,
cleaner V4 dataset of 17,622 sentences that all reported results use.

`aware-v2/` holds that line's training code (`train.py`, `model.py`, `dapt.py`, `losses.py`,
`hpo.py`, `trainer.py`, `evaluate.py`, `predict.py`, `augment_rare_themes.py`), its research
plan, and its project report.

It is here because the path matters: the decision to consolidate themes and clean the data was
an empirical finding, not an initial design choice. For current work, use
[`../models/`](../models/) and [`../experiments/`](../experiments/).
