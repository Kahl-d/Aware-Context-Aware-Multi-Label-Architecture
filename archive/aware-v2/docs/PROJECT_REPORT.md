# AWARE v2: Automated Detection of Community Cultural Wealth Themes in Student Reflective Essays

## Project Report — Master's Thesis Research

**Author:** Khalid Khan
**Institution:** San Francisco State University (SFSU)
**Program:** Master's in Computer Science
**Advisor:** [Advisor Name]
**Date:** February 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [The ALMA Project & Dataset](#3-the-alma-project--dataset)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Model Architecture — AWARE v2](#5-model-architecture--aware-v2)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference & Deployment](#7-inference--deployment)
8. [Research Contributions](#8-research-contributions)
9. [File Reference](#9-file-reference)

---

## 1. Project Overview

### 1.1 Problem Statement

STEM courses at universities increasingly incorporate reflective writing assignments where students describe their personal experiences, motivations, challenges, and cultural backgrounds. These reflective essays contain valuable signals about the cultural capital that students draw upon to navigate higher education. However, manually annotating thousands of essays for cultural capital themes is prohibitively time-consuming — a single trained annotator may take 5–10 minutes per essay, and inter-annotator agreement studies require multiple passes.

This research develops an automated system to detect **Community Cultural Wealth (CCW) themes** at the sentence level within student reflective essays. The system takes a raw essay as input and produces per-sentence multi-label theme annotations, enabling researchers to analyze cultural capital patterns at scale across entire semesters of student writing.

### 1.2 Research Goal

Build a deep learning model that can:

1. **Read a full student essay** and understand the narrative context across sentences.
2. **Classify each sentence** into one or more of 11 cultural capital themes (or identify it as non-themed).
3. **Handle severe class imbalance** — the rarest theme (First Gen) has only 137 examples, while the most common (Navigational) has 9,867 (a 72x ratio).
4. **Generalize to unseen students** — the model must predict themes for students it has never encountered during training, not memorize writing patterns of specific individuals.
5. **Produce research-usable outputs** — annotated essays, theme distributions, and confidence scores that researchers can directly use in their analysis.

### 1.3 Why This Matters

- **For education researchers:** Enables large-scale studies of how students activate cultural capital in STEM. Instead of annotating 100 essays manually, researchers can annotate 10,000 computationally.
- **For equity work:** Identifies which forms of cultural wealth are most prevalent in different student populations, informing support programs and curriculum design.
- **For NLP research:** Advances multi-label document classification in a low-resource, class-imbalanced domain with complex label semantics and co-occurrence patterns.

### 1.4 Key Innovation — Essay-Aware Sentence Classification

Traditional approaches classify each sentence independently. But a sentence like *"My mom always told me I could do it"* could be Familial, Aspirational, or Perseverance depending on what surrounds it. Our approach — **AWARE (Annotation With contextual Awareness of Reflective Essays)** — encodes the entire essay through a transformer, then pools token embeddings into sentence representations, passes them through a bidirectional LSTM to capture sequential context, and classifies each sentence while aware of the full essay narrative.

---

## 2. Theoretical Foundation

### 2.1 Community Cultural Wealth (CCW)

Community Cultural Wealth is a theoretical framework introduced by **Tara Yosso (2005)** that challenges deficit-based views of marginalized communities. Rather than asking "What do these students lack?", CCW asks "What cultural resources do these students bring?" Yosso identified six original forms of capital:

| Capital | Definition | Example in Student Writing |
|---------|-----------|---------------------------|
| **Aspirational** | Ability to maintain hopes and dreams despite real and perceived barriers | *"Even though my family had no money, I always knew I would go to college"* |
| **Navigational** | Skills for maneuvering through social institutions not designed for communities of color | *"I learned how to find tutoring resources and office hours to keep up in class"* |
| **Familial** | Cultural knowledge nurtured among family and community that carries a sense of belonging | *"My grandmother's stories about our homeland gave me strength"* |
| **Social** | Networks of people and community resources that provide instrumental and emotional support | *"My study group helped me survive organic chemistry"* |
| **Resistance** | Knowledge and skills fostered through oppositional behavior that challenges inequality | *"When the professor said I wouldn't make it, I decided to prove them wrong"* |
| **Linguistic** | Intellectual and social skills attained through communication in multiple languages and styles | *(Not used in ALMA — excluded from annotation scheme)* |

### 2.2 Extended Theme Set — The ALMA Project

The ALMA (Assessment of Learning in a Multicultural Academic) Project at SFSU, led by **Tran et al. (2022)**, extended Yosso's framework with additional themes identified through iterative coding of STEM student essays:

| Extended Theme | Definition |
|---------------|-----------|
| **Attainment** | Descriptions of actual accomplishments and achievements in academic settings |
| **Perseverance** | Expressions of persistence through academic difficulties and challenges |
| **Spiritual** | References to faith, spirituality, or religious practices as sources of strength |
| **Filial Piety** | Sense of duty, respect, and obligation to family that motivates academic pursuits |
| **Community Consciousness** | Awareness of and responsibility to one's broader community |
| **First Gen** | Explicit identification as a first-generation college student and its impact |

### 2.3 The 11-Theme Classification Task

The full classification task involves 11 themes (the 5 primary CCW capitals minus Linguistic, plus the 6 extended themes). Each sentence in a student essay receives a multi-label annotation — a single sentence can express multiple forms of cultural capital simultaneously. For example:

> *"[example sentence redacted: student writing is confidential - see data/README.md]octor."*

This sentence simultaneously expresses:
- **Familial** (parents' sacrifice)
- **Aspirational** (wanting to become a doctor)
- **Filial Piety** (desire to make parents proud)
- **Navigational** (understanding education as a pathway)

Sentences with no identified theme are labeled **class_0**.

### 2.4 Prior Computational Work

**TACCTI (Nayak et al., 2020):** The first computational approach to this task used traditional ML (SVM, Random Forest) with TF-IDF features. Key limitations:
- Collapsed 11 themes into 3 super-categories (losing granularity)
- Treated sentences independently (no essay context)
- Small dataset (subset of current data)

**AWARE v1 (Previous attempt):** Our initial deep learning approach using DeBERTa-v3-large achieved a test macro F1 of only 0.24, with severe overfitting (train macro F1 = 0.60) and zero F1 on rare themes (Resistance, Community Consciousness, Filial Piety). Root causes included: oversized encoder (304M params), student data leakage across splits, noisy contrastive loss, and suboptimal learning rate scheduling.

**AWARE v2 (This work)** addresses all identified issues with a rebuilt pipeline.

---

## 3. The ALMA Project & Dataset

### 3.1 Data Source

The dataset comes from the **ALMA Project** at San Francisco State University, which collects 5-minute reflective writing exercises from STEM courses. Students write short essays in response to prompts about their experiences in class, their motivations, and their backgrounds. The data spans:

| Semester | Year | Source |
|----------|------|--------|
| Fall | 2019 | ALMA 2023 |
| Spring | 2020 | ALMA 2023 |
| Fall | 2020 | ALMA 2023 |
| Spring | 2021 | ALMA 2024 |
| Fall | 2021 | ALMA 2024 |
| Spring | 2022 | ALMA 2024 |
| Fall | 2022 | ALMA 2024 |
| Spring | 2025 | New Data (unannotated) |

Courses include: Physics (PHYS), Astronomy (ASTR), Biology (BIOL), Chemistry (CHEM), Engineering (ENGR), Geology (GEOL), Computer Science (CSC), and Mathematics (MATH).

### 3.2 Annotation Process

Each essay was annotated at the sentence level by trained human annotators. For most of the dataset, two independent annotators reviewed each essay. The reconciliation rule used for our training data:

> **If one annotator says YES and the other says NO for a given theme on a given sentence → use YES.**

This permissive rule ensures high recall in the training labels, which is important for rare themes where annotator disagreement often stems from the second annotator missing a subtle reference rather than from a genuine negative.

### 3.3 Raw Dataset Statistics

| Metric | Value |
|--------|-------|
| Total essays collected | 16,227 |
| After cleaning non-essays | 16,148 |
| Total sentences | 112,490 |
| Unique students (alma_id) | 4,020 |
| Courses represented | 27 |
| Semesters covered | 7 |
| Sentences with theme annotations | 26,602 (23.6%) |
| Sentences without themes (class_0) | 85,888 (76.4%) |
| Essays with at least one themed sentence | 9,134 (56.6%) |
| Essays with no themes | 7,014 (43.4%) |

### 3.4 Theme Distribution

The dataset exhibits severe class imbalance across the 11 themes:

| Theme | Sentences | % of Annotated | Essays | Imbalance Ratio |
|-------|-----------|---------------|--------|-----------------|
| Navigational | 9,867 | 37.1% | 4,123 | 1.0x (baseline) |
| Attainment | 6,876 | 25.8% | 4,092 | 1.4x |
| Perseverance | 6,439 | 24.2% | 2,977 | 1.5x |
| Aspirational | 5,993 | 22.5% | 3,357 | 1.6x |
| Familial | 3,083 | 11.6% | 1,843 | 3.2x |
| Social | 1,205 | 4.5% | 743 | 8.2x |
| Resistance | 968 | 3.6% | 424 | 10.2x |
| Spiritual | 873 | 3.3% | 388 | 11.3x |
| Filial Piety | 326 | 1.2% | 167 | 30.3x |
| Community Consciousness | 225 | 0.8% | 106 | 43.9x |
| First Gen | 137 | 0.5% | 79 | 72.0x |

### 3.5 Multi-Label Characteristics

Sentences frequently express multiple themes simultaneously:

| Labels per Sentence | Count | % of Annotated |
|--------------------|-------|---------------|
| 1 label | 19,801 | 74.4% |
| 2 labels | 4,788 | 18.0% |
| 3 labels | 1,623 | 6.1% |
| 4 labels | 278 | 1.0% |
| 5+ labels | 121 | 0.5% |

Key co-occurrence patterns:
- **Aspirational ↔ Attainment**: 2,467 co-occurrences (tightly coupled — aspirations and achievements often described together)
- **Navigational ↔ Aspirational**: 1,779 co-occurrences (navigating systems driven by aspirations)
- **Navigational ↔ Attainment**: 1,941 co-occurrences
- **Perseverance ↔ Resistance**: 699 co-occurrences (resisting barriers through persistence)
- **First Gen ↔ Familial**: 96 out of 137 First Gen sentences also have Familial (almost never appears alone)

---

## 4. Data Processing Pipeline

### 4.1 Overview

The raw ALMA data existed in multiple formats across multiple folders — Excel spreadsheets, reconciled annotation files, and various intermediate processing outputs from different research team members. We built a unified processing pipeline that:

1. Extracts essays from all sources into a common format
2. Cleans text (Unicode, encoding, garbage sentences)
3. Reconciles annotations (multi-annotator → single label set per sentence)
4. Creates modeling-ready data with tracked provenance

### 4.2 Cleaning Steps

**Step 1 — Remove non-essay entries (79 dropped):** Numbers, garbled text, filenames, placeholders, and metadata that were stored as essays but weren't actual student writing.

**Step 2 — Fix Unicode & encoding (4,023 essays):** Non-breaking spaces (`\xa0`), tab characters, Excel control characters (`_x0003_`), garbled smart quotes (`‚Äô`), zero-width spaces, and BOMs. All normalized to clean UTF-8.

**Step 3 — Drop garbage sentences (44 dropped):** Pure punctuation (`:)`, `..`), single characters, and pure numbers that remained after sentence splitting.

**Step 4 — Sequential re-indexing:** All essays assigned clean IDs (`essay_000001` to `essay_016148`) with full traceability to original IDs.

**Step 5 — Field trimming:** Removed `annotator_details` (34.7 MB), `source_files` (3.6 MB), `normalized_text` (~10 MB), and `text_hash` (~1 MB) to reduce file size from 79 MB to 29.9 MB while preserving all information needed for modeling.

### 4.3 Creating the Training Dataset

**The core design challenge:** The raw dataset is 76% class_0 at the sentence level. Using all of it would bury rare themes under an avalanche of non-themed sentences.

**Our approach:**

1. **Keep all class_1 essays** (9,134 essays with at least one themed sentence → 66,435 sentences). These are the core training signal, containing both annotated (26,611) and class_0 (39,824) sentences. The class_0 sentences within themed essays provide important context — the model learns that not every sentence in a themed essay is themed.

2. **Sample class_0 essays as negative examples** (3,010 essays → 19,738 sentences). The model needs to see entire essays with no themes to learn the "fully non-themed" distribution. We sampled at a ratio of 2x the highest theme count (Navigational = 9,867 → target ~19,734 class_0 sentences). This ratio balances providing enough negative examples without overwhelming rare themes.

3. **Artifact cleaning** (119 sentences dropped): Found and removed teacher prompt instructions ("5 minutes of reflective journaling..."), word count artifacts ("165 words"), and URL-only sentences. All were class_0 — no annotated data lost.

### 4.4 Final Model Data

| Metric | Value |
|--------|-------|
| **Total sentences** | **86,054** |
| **Unique essays** | **12,144** |
| Themed essays (class_1) | 9,134 (75.2%) |
| Pure class_0 essays (sampled) | 3,010 (24.8%) |
| Annotated sentences (has theme) | 26,611 (30.9%) |
| class_0 sentences | 59,443 (69.1%) |
| Total words | 1,910,926 |
| Average words/sentence | 22.2 |
| Average sentences/essay | 7.1 |

### 4.5 Train / Validation / Test Splits

Splitting is done **by student (alma_id)**, not by essay or sentence. This prevents data leakage — without student-level splits, the same student's writing style could appear in both training and test sets, inflating metrics.

We use stratified splitting that ensures all 11 themes appear in every split, prioritizing representation of rare themes.

| Split | Essays | Sentences | Students | Leakage |
|-------|--------|-----------|----------|---------|
| **Train** | 9,664 | 68,525 | 3,211 | — |
| **Validation** | 1,179 | 8,387 | 397 | 0 students shared with train |
| **Test** | 1,301 | 9,142 | 412 | 0 students shared with train or val |

**Theme distribution across splits (training set):**

| Theme | Train | Val | Test | Weight |
|-------|-------|-----|------|--------|
| Navigational | 7,879 | 975 | 1,013 | 1.00 |
| Attainment | 5,434 | 675 | 767 | 1.20 |
| Perseverance | 5,163 | 563 | 713 | 1.24 |
| Aspirational | 4,732 | 596 | 665 | 1.29 |
| Familial | 2,432 | 299 | 352 | 1.80 |
| Social | 944 | 135 | 126 | 2.89 |
| Spiritual | 698 | 92 | 83 | 3.36 |
| Resistance | 770 | 93 | 105 | 3.20 |
| Filial Piety | 263 | 25 | 38 | 5.47 |
| Community Consciousness | 170 | 37 | 18 | 6.81 |
| First Gen | 114 | 9 | 14 | 8.31 |
| class_0 | 47,358 | 5,790 | 6,295 | — |

The **Weight** column shows the inverse-sqrt-frequency weight used in the loss function: `weight_i = sqrt(max_count / count_i)`. The rarest theme (First Gen) receives 8.3x the loss weight of the most common (Navigational).

### 4.6 DAPT Corpus

For Domain Adaptive Pre-Training, we use **all available essay text** — both annotated (16,148 essays) and unannotated new data. This provides the widest possible domain coverage for adapting the language model to student reflective writing.

| Corpus | Essays | Size |
|--------|--------|------|
| Annotated (master_data) | 16,148 | ~13 MB |
| Unannotated (new_data) | varies | additional |
| **Total DAPT corpus** | **16,148+** | **13+ MB** |

---

## 5. Model Architecture — AWARE v2

### 5.1 Design Philosophy

The key insight behind AWARE is that **sentence meaning depends on essay context**. A sentence like *"I didn't give up"* is Perseverance if it follows a description of academic struggle, but could be Resistance if it follows a description of discrimination. The model must read the full essay before classifying any individual sentence.

### 5.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL ESSAY TEXT                                │
│  "Sentence 1. Sentence 2. ... Sentence N."                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              DeBERTa-v3-base Encoder (86M params)                │
│                                                                  │
│  Input: tokenized full essay (max 512 tokens)                    │
│  Output: token embeddings [batch, seq_len, 768]                  │
│                                                                  │
│  Optionally initialized from DAPT checkpoint                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sentence Mean Pooling                                │
│                                                                  │
│  For each sentence i:                                            │
│    1. Find token span [start_i, end_i] using offset_mapping      │
│    2. Masked mean of token embeddings within span                │
│    → sentence_embedding_i ∈ R^768                                │
│                                                                  │
│  Output: sentence embeddings [batch, max_sentences, 768]         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              BiLSTM Context Encoder                               │
│                                                                  │
│  2-layer bidirectional LSTM (hidden=256 per direction)           │
│  Input: sentence embeddings [batch, max_sent, 768]               │
│  LSTM output: [batch, max_sent, 512]                             │
│  Linear projection: 512 → 768                                    │
│  Output: context-aware embeddings [batch, max_sent, 768]         │
│                                                                  │
│  Each sentence now encodes information from all                   │
│  surrounding sentences in the essay                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Classification Head                                  │
│                                                                  │
│  Dropout(0.3) → LayerNorm(768) → Linear(768, 11)                │
│                                                                  │
│  Output: logits [batch, max_sentences, 11]                       │
│  → Sigmoid → per-theme probabilities                             │
│  → Per-theme optimized thresholds → binary predictions           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Component Details

**DeBERTa-v3-base (86M parameters):**
- Uses Disentangled Attention (separate content and position embeddings)
- Replaced Absolute Position with Enhanced Mask Decoder (ELECTRA-style)
- v3 uses ELECTRA-style pre-training (more efficient than MLM)
- 12 layers, 12 attention heads, 768 hidden size
- Max sequence length: 512 tokens (covers most essays — avg essay is ~160 tokens)
- Chosen over DeBERTa-v3-large (304M) to reduce overfitting risk

**Sentence Mean Pooling:**
- Maps character-level sentence boundaries to token-level spans using `offset_mapping` from the tokenizer
- Computes masked mean of token embeddings within each sentence span
- Handles variable-length essays (up to 32 sentences)
- Truncated sentences (beyond 512 tokens) are excluded via sentence mask

**BiLSTM Context Encoder:**
- 2-layer bidirectional LSTM with 256 hidden units per direction
- Uses `pack_padded_sequence` for efficient variable-length processing
- Linear projection (512 → 768) maps back to encoder dimension
- Adds sequential context: each sentence embedding now carries information about its position in the essay narrative

**Classification Head:**
- Dropout (0.3) for regularization
- LayerNorm for training stability (prevents internal covariate shift)
- Single linear layer (768 → 11) for multi-label prediction
- No sigmoid here — applied separately during loss computation and inference

### 5.4 Model Size

| Component | Parameters |
|-----------|-----------|
| DeBERTa-v3-base encoder | 183.8M |
| BiLSTM (2-layer, bidirectional, 256 hidden) | 3.2M |
| Linear projection (512 → 768) | 0.4M |
| Classification head (768 → 11) | 0.008M |
| LayerNorm | 0.002M |
| **Total** | **~187.9M** |
| Trainable in Phase 1 (encoder frozen) | ~4.1M |
| Trainable in Phase 2 (all) | ~187.9M |

### 5.5 Why Not Simpler Approaches?

| Alternative | Why Not |
|------------|---------|
| Classify sentences independently (no essay context) | Loses critical context — same sentence text can map to different themes depending on surrounding narrative |
| Sentence-level transformer (e.g., fine-tune BERT on individual sentences) | Misses co-occurrence patterns and narrative structure that the BiLSTM captures |
| Document-level classification (assign themes to entire essay) | Too coarse — researchers need sentence-level granularity for their analysis |
| Simple mean pooling of essay (no BiLSTM) | BiLSTM adds sequential structure — a sentence about "struggle" followed by "success" is different from "success" followed by "struggle" |

---

## 6. Training Pipeline

### 6.1 Domain Adaptive Pre-Training (DAPT)

Before fine-tuning on the classification task, we adapt DeBERTa's language model to the domain of student reflective writing using Masked Language Modeling (MLM) on our full essay corpus. Following **Gururangan et al. (2020)**, this helps the model understand domain-specific vocabulary and writing patterns.

| DAPT Setting | Value |
|-------------|-------|
| Corpus | All available essays (16,148+ essays, 13+ MB) |
| Task | Masked Language Modeling (MLM) |
| Masking probability | 15% |
| Epochs | 5 |
| Batch size | 16 |
| Learning rate | 5e-5 |
| Warmup | 10% of steps |
| Precision | FP16 |

The DAPT encoder is saved and then used as the initialization point for the AWARE fine-tuning.

### 6.2 Two-Phase Fine-Tuning

Training proceeds in two phases to prevent catastrophic forgetting of the pre-trained encoder while allowing the randomly initialized BiLSTM and classification head to find a good starting point.

#### Phase 1: Frozen Encoder (5 epochs)

- **DeBERTa encoder weights are frozen** — no gradients flow back through the transformer
- Only the BiLSTM, projection layer, classification head, and LayerNorm are trained
- Learning rate: 1e-4 (decoder LR only)
- This allows the randomly initialized components to learn useful representations before the encoder starts adapting
- Effective batch size: 8 × 4 (gradient accumulation) = 32

#### Phase 2: Full Fine-Tuning (up to 15 epochs)

- **All parameters are unfrozen** — the encoder adapts to our task
- Differential learning rates:
  - Encoder: 2e-6 (= 2e-5 × 0.1 scale factor) — small updates to preserve pre-trained knowledge
  - Decoder (BiLSTM + head): 1e-4 — continues learning at full speed
- Early stopping with patience of 7 epochs on validation macro F1
- Effective batch size: 8 × 4 = 32

### 6.3 Learning Rate Schedule

Both phases use **linear warmup + cosine decay**, the standard schedule for transformer fine-tuning:

```
LR
 ↑
 │    /\
 │   /  \
 │  /    \_____
 │ /            \____
 │/                   \____
 └──────────────────────────→ Steps
   warmup   cosine decay
   (6%)
```

This replaced OneCycleLR from v1, which was causing instability during phase transitions.

### 6.4 Loss Function — Asymmetric Loss (ASL)

We use **Asymmetric Loss (Ridnik et al., 2021)** instead of standard Binary Cross-Entropy. ASL is specifically designed for multi-label classification with class imbalance:

```
L_ASL = -y · (1-p)^γ+ · log(p) - (1-y) · p_m^γ- · log(1-p_m)
```

Where:
- `γ+` = 1.0 (positive focusing — moderate hard positive mining)
- `γ-` = 4.0 (negative focusing — aggressively down-weight easy negatives)
- `p_m = max(p - clip, 0)` with `clip` = 0.05 (probability shifting — prevents very confident negatives from contributing zero loss)

**Why ASL over BCE:**
- With 69% class_0, standard BCE is dominated by negative gradients
- ASL's asymmetric gamma values let the model focus on learning positive examples
- The clip parameter prevents "dead" gradients from very easy negatives
- Combined with per-theme weights (inverse sqrt frequency), this addresses both inter-theme imbalance and the class_0 dominance

**Per-theme weights:**

| Theme | Training Sentences | Weight |
|-------|-------------------|--------|
| Navigational | 7,879 | 1.00 |
| Attainment | 5,434 | 1.20 |
| Perseverance | 5,163 | 1.24 |
| Aspirational | 4,732 | 1.29 |
| Familial | 2,432 | 1.80 |
| Social | 944 | 2.89 |
| Resistance | 770 | 3.20 |
| Spiritual | 698 | 3.36 |
| Filial Piety | 263 | 5.47 |
| Community Consciousness | 170 | 6.81 |
| First Gen | 114 | 8.31 |

Label smoothing of 0.05 is also applied to prevent overconfident predictions.

### 6.5 Data Augmentation — AEDA

During training, we apply **AEDA (An Easier Data Augmentation, Karimi et al., 2021)** — random punctuation insertion at word boundaries with 30% probability. This is simpler and less destructive than synonym replacement or back-translation, and helps regularize the model without changing semantic content.

Example:
- Original: *"My family always supported me in school"*
- AEDA: *"My family . always supported me ; in school"*

### 6.6 Weighted Random Sampling

Training uses weighted random sampling at the essay level. Essays containing rare themes are sampled more frequently:

```
weight(essay) = max over themes of sqrt(total_themed_essays / count(theme))
```

Pure class_0 essays receive weight 0.5 (down-sampled). This ensures that essays with Filial Piety, Community Consciousness, and First Gen are seen more often during training.

### 6.7 Per-Theme Threshold Optimization

After training, the default 0.5 sigmoid threshold is replaced with per-theme optimized thresholds. We grid-search thresholds from 0.15 to 0.65 on the validation set, optimizing F1 per theme. Rare themes typically get lower thresholds (more permissive — higher recall) because missing a rare theme is more costly than a false positive.

### 6.8 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Macro F1** | Average F1 across all 11 themes (primary metric — treats all themes equally) |
| **Micro F1** | Global F1 (weighted by frequency — dominated by common themes) |
| **Per-theme F1** | Individual F1 for each of the 11 themes (critical for rare themes) |
| **Per-theme Precision & Recall** | Detailed error analysis |
| **Bootstrap 95% CI** | Confidence intervals via 1,000 bootstrap samples |
| **Confusion Analysis** | Which themes get confused with which others |

### 6.9 Regularization Summary

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Dropout | 0.3 | Prevent co-adaptation of features |
| Weight decay | 0.01 | L2 regularization on all parameters |
| Label smoothing | 0.05 | Prevent overconfident predictions |
| Gradient clipping | max_norm=1.0 | Training stability |
| AEDA augmentation | 30% probability | Data augmentation |
| Early stopping | Patience 5 (P1) / 7 (P2) | Prevent overfitting |
| DeBERTa-v3-base (not large) | 86M vs 304M | Reduce model capacity to match data size |
| Two-phase training | Freeze → unfreeze | Prevent catastrophic forgetting |
| Differential LR | 10x lower for encoder | Preserve pre-trained knowledge |

### 6.10 Hyperparameter Optimization

Optuna-based HPO is available to tune key hyperparameters:

- Encoder LR: [5e-6, 5e-5] (log scale)
- Decoder LR: [3e-5, 3e-4] (log scale)
- Dropout: [0.1, 0.4]
- ASL gamma_neg: [2.0, 6.0]
- ASL clip: [0.0, 0.1]
- Weight decay: [0.001, 0.05]
- Phase 2 encoder LR scale: [0.05, 0.3]
- Label smoothing: [0.0, 0.1]

### 6.11 Complete Training Configuration

```yaml
model:
  encoder_name: microsoft/deberta-v3-base
  hidden_size: 768
  lstm_hidden: 256
  lstm_layers: 2
  max_sentences: 32
  max_seq_length: 512
  num_labels: 11
  dropout: 0.3

loss:
  asl_gamma_pos: 1.0
  asl_gamma_neg: 4.0
  asl_clip: 0.05
  label_smoothing: 0.05

training:
  phase1_epochs: 5
  phase2_epochs: 15
  batch_size: 8
  gradient_accumulation: 4
  encoder_lr: 2.0e-5
  decoder_lr: 1.0e-4
  phase2_encoder_lr_scale: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_ratio: 0.06
  early_stopping_patience: 5
  fp16: true
```

---

## 7. Inference & Deployment

### 7.1 Single Essay Prediction

The trained model can annotate a single essay:

```bash
python3 scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
    --text "I came to this country when I was young. My family always believed in education.
    Even when things were hard, my mom told me to keep going. Now I'm the first in my
    family to go to college."
```

Output:
```
[Navigational, Familial]
I came to this country when I was young.

[Familial, Aspirational]
My family always believed in education.

[Perseverance, Familial]
Even when things were hard, my mom told me to keep going.

[First Gen, Attainment, Aspirational]
Now I'm the first in my family to go to college.
```

### 7.2 Batch Prediction

For research use, the model can process entire datasets:

```bash
python3 scripts/predict.py --model_dir results/full/ --config configs/full.yaml \
    --input_file essays.csv --output_file annotated_essays.xlsx
```

Input: CSV/Excel with an essay text column.
Output: Excel file with per-sentence theme predictions and probability scores for all 11 themes.

### 7.3 HPC Deployment

The `cModels/` folder is self-contained and designed for HPC deployment:

```bash
# On HPC cluster:
# 1. Upload cModels/ folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run DAPT (4 hours on single GPU)
sbatch run_dapt.slurm

# 4. Run training (8 hours on single GPU)
sbatch run_train.slurm
```

---

## 8. Research Contributions

### 8.1 Technical Contributions

1. **Essay-aware sentence classification (AWARE):** A novel architecture that classifies sentences in the context of their full essay narrative, using DeBERTa + mean pooling + BiLSTM.

2. **Domain Adaptive Pre-Training for educational text:** Application of DAPT (Gururangan et al., 2020) to student reflective writing, adapting a general-purpose language model to the specific domain.

3. **Handling extreme class imbalance in multi-label NLP:** A comprehensive strategy combining ASL, inverse-sqrt frequency weights, per-theme threshold optimization, and weighted sampling to address a 72x imbalance ratio.

4. **Student-level data splitting:** Prevention of data leakage through student-level (not essay-level) stratified splits, ensuring evaluation reflects true generalization to unseen students.

### 8.2 Applied Contributions

1. **Scalable theme annotation for education research:** Enabling automated annotation of thousands of student essays that would be impractical to annotate manually.

2. **Complete, reproducible pipeline:** From raw data cleaning through model training to batch inference, fully documented and deployable on university HPC clusters.

3. **Comprehensive data processing documentation:** Every cleaning step, design decision, and statistical summary is documented with rationale, enabling full reproducibility.

### 8.3 What Makes This Work Unique

- **Granularity:** Sentence-level classification (not document or paragraph level) with 11 specific themes (not collapsed super-categories)
- **Context awareness:** Essay-level encoding before sentence-level classification — previous computational work (TACCTI) treated sentences independently
- **Full 11-theme detection:** Including rare themes that previous work collapsed or ignored (First Gen, Community Consciousness, Filial Piety)
- **Real-world deployment path:** Not just a model, but a complete system with single-essay and batch prediction capabilities

---

## 9. File Reference

### 9.1 Project Structure

```
cModels/
├── configs/
│   ├── toy.yaml              # Local testing config (2+3 epochs, batch 2)
│   └── full.yaml             # Full HPC training config (5+15 epochs, batch 8)
├── data/
│   ├── train_data.pkl        # Training split (9,664 essays, 68,525 sentences)
│   ├── val_data.pkl          # Validation split (1,179 essays, 8,387 sentences)
│   ├── test_data.pkl         # Test split (1,301 essays, 9,142 sentences)
│   ├── toy_data.pkl          # Toy subset (100 train + 33 val essays)
│   ├── dapt_corpus.txt       # DAPT MLM training text (16,148+ essays, 13+ MB)
│   └── splits_stats.json     # Split statistics and theme weights
├── scripts/
│   ├── config.py             # Configuration dataclasses + YAML loading
│   ├── model.py              # AWAREModel architecture
│   ├── dataset.py            # Essay-level Dataset + DataLoader + AEDA
│   ├── losses.py             # Asymmetric Loss with per-theme weights
│   ├── metrics.py            # F1, threshold optimization, bootstrap CI
│   ├── trainer.py            # Two-phase training loop
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation with detailed per-theme reports
│   ├── predict.py            # Single essay + batch inference
│   ├── dapt.py               # Domain Adaptive Pre-Training (MLM)
│   ├── hpo.py                # Optuna hyperparameter optimization
│   ├── run_pipeline.py       # Full pipeline orchestration
│   ├── prepare_data.py       # Data → essay format + stratified splits
│   └── test_components.py    # Component verification tests (12/12 passing)
├── docs/
│   └── PROJECT_REPORT.md     # This document
├── results/                  # Training outputs (checkpoints, metrics, thresholds)
├── run_dapt.slurm            # HPC job script for DAPT
├── run_train.slurm           # HPC job script for training + evaluation
├── requirements.txt          # Python dependencies
└── README.md                 # Quick start guide
```

### 9.2 Data Files (Source)

```
Data_for_modeling/
├── model_data.pkl            # Final training data (86,054 sentences, 12,144 essays)
├── master_data.pkl           # Full dataset with metadata (16,148 essays)
├── class1_essays.pkl         # Themed essays (9,134)
├── class0_essays.pkl         # Non-themed essays (7,014)
└── PROCESSING_LOG.md         # Complete processing documentation (10 steps)
```

### 9.3 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.0 | Deep learning framework |
| transformers | ≥ 4.36 | DeBERTa-v3, tokenizers, training utilities |
| datasets | ≥ 2.16 | HuggingFace datasets for DAPT |
| numpy | ≥ 1.24 | Numerical computing |
| pandas | ≥ 2.0 | Data manipulation for batch prediction |
| scikit-learn | ≥ 1.3 | Metrics utilities |
| pyyaml | ≥ 6.0 | Config file parsing |
| optuna | ≥ 3.4 | Hyperparameter optimization |
| tqdm | ≥ 4.65 | Progress bars |

---

## References

- Gururangan, S., et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *ACL*.
- He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *ICLR*.
- He, P., et al. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *ICLR*.
- Karimi, A., et al. (2021). AEDA: An Easier Data Augmentation Technique for Text Classification. *EMNLP Findings*.
- Nayak, A., et al. (2020). TACCTI: Thematic Analysis of Cultural Capital through Text and Images. *Computing Research Repository*.
- Ridnik, T., et al. (2021). Asymmetric Loss for Multi-Label Classification. *ICCV*.
- Tran, M.C., et al. (2022). Reflective Writing as a Tool for Examining Student Cultural Capital in STEM. *Education Sciences*.
- Yosso, T.J. (2005). Whose Culture Has Capital? A Critical Race Theory Discussion of Community Cultural Wealth. *Race Ethnicity and Education*, 8(1), 69–91.
