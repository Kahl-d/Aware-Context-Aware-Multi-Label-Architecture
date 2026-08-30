# ALMA Research Dashboard

Two complementary interfaces built for the ALMA research team. Together they close the loop of
the thesis: the pipeline produces clean data, AWARE learns from it, and the dashboard puts both
the data and the model's predictions in front of the researchers who need them.

## `corpus-explorer/`

React 19 + Vite + Recharts. Data exploration and analytics over the corpus.

- **Data view** — spreadsheet over all essays; full-text search; filter by theme, semester,
  year, course, and data source. Expanding a row shows the essay with sentence-level
  annotations colour-coded by theme, which is the view that matters for annotation quality
  review.
- **Stats view** — theme distributions, the theme co-occurrence heatmap, temporal coverage
  (which exposes the incomplete-annotation problem directly), annotation density, and training
  readiness metrics.

## `annotation-workbench/`

React frontend + FastAPI inference server. Adds **live model inference** to exploration, so
AWARE can act as an automated annotator over the 1,388 uncoded essays.

Predictions appear alongside human annotations with calibrated confidence scores, tagged by
source so a researcher can filter to human-only, model-only, or both. The interface never
disguises a prediction as a human judgment. Corrected labels feed the next training round.

## Running

Each app has its own README and `package.json`. Both expect corpus data that is **not** shipped
with this repository — see [`../data/README.md`](../data/README.md). Generated data files were
removed before publication; regenerate them locally from the ALMA source.

> Model predictions carry the same confidentiality obligations as the essays themselves.
