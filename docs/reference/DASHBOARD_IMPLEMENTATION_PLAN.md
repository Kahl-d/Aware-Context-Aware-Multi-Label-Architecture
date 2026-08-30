# ALMA Research Dashboard — Implementation Plan

## Overview

A 4-page research dashboard for the AWARE thesis: Data Explorer, Model Inference, Research & Exploration, and AWARE Paper (interactive research presentation).

**Stack:** React 19 + Vite + TypeScript (frontend) | Python FastAPI (inference backend) | Pre-computed JSON (data)

---

## System Architecture

```
Browser (Client)
  React 19 + Vite + TypeScript + Tailwind CSS
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐
  │   Data   │ │  Model   │ │ Research │ │   AWARE Paper    │
  │ Explorer │ │Inference │ │& Explore │ │  (Interactive)   │
  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘
       │            │            │                  │
  ┌────┴────────────┴────────────┴──────────────────┴──────────┐
  │              Shared State (Zustand Store)                   │
  └────┬────────────┬──────────────────────────────────────────┘
       │            │
  Static JSON   API calls
  (public/)     (fetch)
       │            │
       │            ▼
       │   ┌─────────────────────────────────────┐
       │   │   Python FastAPI Inference Server    │
       │   │   POST /api/infer/single             │
       │   │   POST /api/infer/batch              │
       │   │   GET  /api/models                   │
       │   │   GET  /api/health                   │
       │   │   Loads: DeBERTa-large (1.7GB)       │
       │   │          DeBERTa-base (718MB)         │
       │   │          TF-IDF pipeline              │
       │   └─────────────────────────────────────┘
       ▼
  /public/data/*.json
  Pre-computed static data files
```

**Key decisions:**
- Frontend serves pre-computed JSON for Pages 1, 3, 4. No database.
- Python FastAPI server ONLY needed for Page 2 (inference) and "Analyze" buttons.
- No authentication. Pure SPA + local API.
- Anthropic/Claude-like design: clean, minimal, research-grade.

---

## The 4 Pages

### PAGE 1: DATA EXPLORER
Spreadsheet-like interface over ALL ALMA data with full metadata tracking.

**Features:**
- TanStack Table with virtual scrolling (handles 20K+ rows)
- Row grouping by essay_id with expand/collapse
- Color-coded theme badges (ThemeBadge component)
- Filters: Dataset version, Annotated/Unannotated, Split (train/val/test), Theme, Course, Semester, Year, Prompt, Coder
- Full-text search (essay_id, alma_id, sentence content)
- Essay detail panel: click row → see all sentences with labels, metadata, confidence
- "Analyze" button on unannotated rows → runs model inference
- Export filtered data as CSV
- Tags on each row: annotated/unannotated, dataset version, split assignment, dropped reason

### PAGE 2: MODEL INFERENCE
Two inference modes using trained AWARE models.

**Feature 1: Single Essay Analysis**
- Textarea input (essay or single sentence)
- Model selection: Large v4 (F1=0.494), Base (F1=0.474), TF-IDF (F1=0.378)
- Live sentence count as user types
- Results: each sentence with 8 confidence bars, threshold markers, predicted theme badges

**Feature 2: Batch CSV Upload**
- Upload CSV template (columns: essay_id, essay_text)
- Progress bar via SSE
- Download results CSV with per-sentence predictions

**Backend:** FastAPI server with /api/infer/single and /api/infer/batch endpoints.

### PAGE 3: RESEARCH & EXPLORATION
Interactive data analysis with flexible subsetting and visualization.

**Features:**
- Subset builder (reuses Page 1 filters)
- Compare mode: A/B subset comparison
- 7 visualization types:
  1. Theme distribution (bar chart)
  2. Co-occurrence heatmap (8x8)
  3. Multi-label cardinality histogram
  4. Temporal trends (line chart by semester/year)
  5. Course breakdown (stacked bars)
  6. Sentence length distribution
  7. Embedding visualization (UMAP/PCA scatter, Canvas-based)
- Export subsets as CSV
- "Analyze All Unannotated" batch button

### PAGE 4: AWARE PAPER (Interactive Research Presentation)
Tab-based (NOT scroll) interactive thesis presentation for defense and conferences.

**6 Tabs:**
1. **Abstract** — Key numbers with animated counters, problem statement
2. **Data** — Pipeline visualization (V1→V4), interactive stats, embedding space
3. **Methodology** — Interactive architecture diagram (SVG, clickable components), 3-phase training timeline, loss breakdown, regularization stack
4. **Engineering** — 7 failure modes (v3→v4), before/after comparisons, training curves
5. **Results** (MOST IMPORTANT) — Model comparison table, per-theme F1 chart, single-theme vs multi-theme finding, threshold optimization waterfall, overfitting analysis, embedding before/after
6. **Conclusions** — RQ answer cards, 5 contributions, future directions

---

## Implementation Phases

### Phase 0: Project Setup (Day 1)
- Initialize Vite + React 19 + TypeScript
- Install: TanStack Table, Recharts, Zustand, React Router, Tailwind CSS
- Set up folder structure, routing, Tailwind theme
- Configure Anthropic-like design system (Inter font, clean palette)

### Phase 1: Data Preparation Pipeline (Days 2-3)
- Python scripts to convert all CSVs → unified JSON
- Compute version membership tags (which sentences in which V)
- Extract train/val/test split assignments from pkl files
- Merge unannotated data with metadata
- Pre-compute embedding coordinates from .npz files
- Generate all static JSON files
- Validate: counts match thesis exactly (17,622 V4, 2,636 essays, etc.)

### Phase 2: Core UI Shell + Data Explorer (Days 4-7)
- App shell (nav, routing, layout)
- FilterBar with all dropdowns
- DataTable with virtual scrolling + essay grouping
- ThemeBadge component and color system
- EssayDetailPanel (slide-out)
- Search, sort, CSV export

### Phase 3: Research & Exploration Page (Days 8-10)
- SubsetBuilder (A/B compare mode)
- All 7 chart components (Recharts)
- Canvas-based embedding scatter plot
- Charts reactive to filter state

### Phase 4: Python Inference Server (Days 11-14)
- FastAPI with CORS
- Model loading (lazy: large on startup, base on first request)
- Sentence segmentation (spaCy)
- Single + batch inference endpoints
- TF-IDF baseline
- Test against README.md examples

### Phase 5: Model Inference Page (Days 15-17)
- Single essay analysis UI
- Confidence bar visualization
- Batch upload with progress
- Connect to inference API
- "Analyze" buttons on Pages 1 and 3

### Phase 6: AWARE Paper Page (Days 18-22)
- Tab navigation for 6 sections
- Interactive architecture diagram (SVG)
- Training timeline, failure mode cards
- All results charts (interactive, hover for CIs)
- Animated counters, accordions, dropdowns

### Phase 7: Polish & Integration (Days 23-25)
- Responsive design
- Cross-page navigation (click theme in results → filtered data explorer)
- Performance optimization (code splitting, memoization)
- Final number verification against thesis

---

## Tech Stack Summary

| Layer | Technology | Reason |
|-------|-----------|--------|
| Frontend | React 19 + Vite + TypeScript | Modern, fast, type-safe |
| Styling | Tailwind CSS (v4) | Claude/Anthropic-like aesthetic |
| State | Zustand | Lightweight, no boilerplate |
| Routing | React Router v7 | URL-encoded filter state |
| Table | TanStack Table v8 + react-virtual | Virtual scrolling for 20K rows |
| Charts | Recharts | React-native, declarative |
| Scatter | Custom HTML5 Canvas | 18K+ points too many for SVG |
| Backend | Python FastAPI | Model inference server |
| ML | PyTorch + Transformers | DeBERTa inference |
| Fonts | Inter (body), JetBrains Mono (numbers) | Clean research typography |
