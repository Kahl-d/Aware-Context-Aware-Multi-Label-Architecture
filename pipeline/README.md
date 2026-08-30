# Data processing pipeline

Raw ALMA essay files → the V4 dataset the model trains on. Run the stages in order.

| Stage | Directory | What happens |
|---|---|---|
| 1 | `1_extraction/` | Assembly and normalization; essay-level coding is projected to sentences via excerpt-to-sentence label propagation. |
| 2 | `2_processing/` | **Embedding-based semantic cleaning** — removes sentences whose embeddings contradict their assigned labels. |
| 3 | `3_final_datasets/` | Builds V1→V4 and the distribution plots. |

## The cleaning step is the interesting one

Stage 2 removed **1,705 sentences (8.6%)** without any manual inspection, by comparing each
sentence's embedding against its theme's centroid and dropping the contradictions. Hand-checking
17,000 sentences was not feasible; this made the cleaning tractable and reproducible.

Label propagation in stage 1 is a known noise source: annotators coded *excerpts*, and those
labels are propagated to the sentences within them, so a sentence can inherit a label it does
not itself carry. Thesis §7.4.3 discusses the consequences.

## Theme consolidation

The original coding protocol used **11** themes. Stage 3 consolidates to **8**, merging
categories that were not separable in embedding space and could not be reliably distinguished
by annotators either. Thesis Table 3.9 records each decision and its rationale.

Source data is not included — see [`../data/README.md`](../data/README.md).
