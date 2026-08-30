# UI Wireframes & Component Design

## Design System

**Aesthetic:** Claude/Anthropic-like — clean, minimal, research-grade
- Background: white (#ffffff) / gray-50 (#f9fafb)
- Text: gray-900 for headings, gray-700 for body
- Borders: gray-200, subtle
- Fonts: Inter (body), JetBrains Mono (numbers/metrics)
- No excessive shadows. Generous whitespace. Clean lines.
- Professional, minimalistic — no random titles or decorative elements.

## Theme Color System

```
Attainment:       #7c3aed (violet)
Aspirational:     #2563eb (blue)
Navigational:     #0891b2 (cyan)
Resistance:       #dc2626 (red)
Perseverance:     #ea580c (orange)
Social:           #16a34a (green)
Spiritual:        #a855f7 (purple)
Familial_Capital: #ca8a04 (amber)
Class_0:          #6b7280 (gray)
```

Abbreviations: ATT, ASP, NAV, RES, PER, SOC, SPI, FAM, C0

---

## PAGE 1: DATA EXPLORER

```
┌──────────────────────────────────────────────────────────────────┐
│  [AWARE]    Data Explorer  │  Inference  │  Explore  │  Paper   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Filter Bar ────────────────────────────────────────────────┐ │
│  │ Dataset: [v4 ▼]  Status: [All ▼]   Split: [All ▼]          │ │
│  │ Theme:   [Any ▼]  Course: [All ▼]  Semester: [All ▼]       │ │
│  │ Year:    [All ▼]  Prompt: [All ▼]  Coder: [All ▼]          │ │
│  │ 🔍 Search: [essay_id, alma_id, or text...           ]      │ │
│  │                                                             │ │
│  │ Showing 17,622 of 20,724 sentences    [Export CSV] [Reset]  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Data Table (virtual scroll) ───────────────────────────────┐ │
│  │ ▶ Essay │ #  │ Sentence                │ Themes    │ Tags   │ │
│  │ ────────────────────────────────────────────────────────────  │ │
│  │ ▶ 042  │  1 │ When life gets chall... │ ─         │ v4 trn │ │
│  │   042  │  2 │ I think that my fam...  │ FAM NAV   │ v4 trn │ │
│  │   042  │  3 │ For example, if I...    │ PER       │ v4 trn │ │
│  │ ▶ 043  │  1 │ [essay text]          │ FAM ASP   │ v4 val │ │
│  │ ▶ U-01 │  ─ │ [unannotated essay]     │ [Analyze] │ unan   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Essay Detail Panel (slide-out on click) ───────────────────┐ │
│  │ Essay 042 │ ASTR116 │ Spring 2025 │ Coder: merged          │ │
│  │ Prompt: "What do I do when life gets challenging?"          │ │
│  │ Split: train │ Versions: v1, v2, v4 │ 7 sentences          │ │
│  │                                                             │ │
│  │  1. "When life gets challenging, I try to..."               │ │
│  │     [Class_0]                                               │ │
│  │                                                             │ │
│  │  2. "I think that my family has always..."                  │ │
│  │     [Familial_Capital] [Navigational]                       │ │
│  │     Conf: FAM 0.81  NAV 0.73                               │ │
│  │                                                             │ │
│  │  3. "For example, if I don't understand..."                 │ │
│  │     [Perseverance]                                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Key components:**
- FilterBar: Multi-select dropdowns, search with 300ms debounce
- DataTable: TanStack Table v8 + @tanstack/react-virtual
- ThemeBadge: Colored pill `<span style={{bg: THEME_COLORS[theme]}}>{abbrev}</span>`
- EssayDetailPanel: Slide-out panel showing all sentences with labels
- AnalyzeButton: For unannotated rows, triggers inference API

---

## PAGE 2: MODEL INFERENCE

```
┌──────────────────────────────────────────────────────────────────┐
│  [AWARE]    Data Explorer  │  Inference  │  Explore  │  Paper   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Model Selection ───────────────────────────────────────────┐ │
│  │ (●) AWARE Large v4  F1=0.494  360M params                  │ │
│  │ ( ) AWARE Base       F1=0.474  125M params                  │ │
│  │ ( ) TF-IDF Baseline  F1=0.378  10K features                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ [Single Essay] │ [Batch CSV] ──────────────────────────────┐ │
│  │                                                              │ │
│  │  Paste an essay or single sentence:                         │ │
│  │  ┌──────────────────────────────────────────────────────┐   │ │
│  │  │                                                      │   │ │
│  │  │ I want to get my degree to make my family proud.     │   │ │
│  │  │ [sentence text redacted - see data/README.md]     │   │ │
│  │  │                                                      │   │ │
│  │  └──────────────────────────────────────────────────────┘   │ │
│  │  4 sentences detected                        [Analyze ▶]   │ │
│  │                                                              │ │
│  │  ── Results ──────────────────────────────────────────────  │ │
│  │                                                              │ │
│  │  Sentence 1: "I want to get my degree to make my..."        │ │
│  │  ┌────────────────────────────────────────────────────┐     │ │
│  │  │ ATT  ████████████░░░░░░░░  0.72  ✓  (t=0.05)     │     │ │
│  │  │ ASP  ██████████████████░░  0.91  ✓  (t=0.23)     │     │ │
│  │  │ NAV  ████░░░░░░░░░░░░░░░  0.18     (t=0.31)     │     │ │
│  │  │ RES  ██░░░░░░░░░░░░░░░░░  0.05     (t=0.09)     │     │ │
│  │  │ PER  █████░░░░░░░░░░░░░░  0.22     (t=0.35)     │     │ │
│  │  │ SOC  ████████░░░░░░░░░░░  0.42  ✓  (t=0.22)     │     │ │
│  │  │ SPI  ██░░░░░░░░░░░░░░░░░  0.08  ✓  (t=0.053)    │     │ │
│  │  │ FAM  ██████████████░░░░░  0.68  ✓  (t=0.14)     │     │ │
│  │  └────────────────────────────────────────────────────┘     │ │
│  │                                                              │ │
│  │  Sentence 2: "[sentence text redacted]"                        │ │
│  │  [confidence bars...]                                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Batch CSV tab:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Expected CSV format:                                           │
│  ┌──────────────────────────────────────────────┐              │
│  │ essay_id  │ essay_text                        │              │
│  │ 1         │ "I want to get my degree..."      │              │
│  │ 2         │ "When life gets challenging..."    │              │
│  └──────────────────────────────────────────────┘              │
│  [Download template CSV]                                        │
│                                                                 │
│  ┌─ Drop CSV here or click to browse ──────────────────────┐   │
│  │                                                          │   │
│  │        📄 Drop your CSV file here                        │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ── Processing ──                                              │
│  [████████████████████░░░░░░░░░░] 67/100 essays  67%           │
│                                                                 │
│  ── Summary ──                                                 │
│  100 essays processed, 742 sentences classified                 │
│  Theme distribution: NAV 234, ASP 189, PER 98, ...            │
│  [Download Results CSV]                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## PAGE 3: RESEARCH & EXPLORATION

```
┌──────────────────────────────────────────────────────────────────┐
│  [AWARE]    Data Explorer  │  Inference  │  Explore  │  Paper   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Subset Builder ────────────────────────────────────────────┐ │
│  │ [Same filters as Data Explorer]                              │ │
│  │ ☐ Compare mode: Subset A (current) │ Subset B               │ │
│  │ A: 2,158 sentences (122 essays)                             │ │
│  │ [Export CSV] [Analyze Unannotated ▶]                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ Theme Distribution   │  │ Co-occurrence        │              │
│  │ ┌─────────┐         │  │ Heatmap (8x8)        │              │
│  │ │ NAV ███████│4,245  │  │ ┌─────────────────┐  │              │
│  │ │ ASP █████│  2,658  │  │ │  colorized grid  │  │              │
│  │ │ PER ███│    1,475  │  │ │  hover = count   │  │              │
│  │ │ ...     │         │  │ └─────────────────┘  │              │
│  │ └─────────┘         │  │                      │              │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ Label Cardinality    │  │ Temporal Trends      │              │
│  │ 0 themes: 50.0%     │  │ ──── NAV ──── ASP    │              │
│  │ 1 theme:  35.7%     │  │ theme % by semester   │              │
│  │ 2+:       14.3%     │  │ /year line chart      │              │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ Course Breakdown     │  │ Sentence Length      │              │
│  │ stacked bars by      │  │ histogram            │              │
│  │ course × theme       │  │                      │              │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                  │
│  ┌──────────────────────────────────────────────┐              │
│  │ Embedding Visualization (Canvas)              │              │
│  │ [UMAP │ PCA]  Color by: [Theme ▼]            │              │
│  │                                               │              │
│  │    scatter plot (18K+ points)                  │              │
│  │    zoom + pan + hover tooltips                 │              │
│  │                                               │              │
│  └──────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

---

## PAGE 4: AWARE PAPER (Interactive Research Presentation)

```
┌──────────────────────────────────────────────────────────────────┐
│  AWARE                                                           │
│  Automated Detection of Community Cultural Wealth Themes         │
│  in Student Reflective Essays Using Domain-Adapted DeBERTa       │
│                                                                  │
│  [Abstract] [Data] [Methodology] [Engineering] [Results] [Conc.] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ── Currently: Results Tab ──────────────────────────────────── │
│                                                                  │
│  ┌ Overall Performance ────────────────────────────────────────┐ │
│  │ Model             │ Macro-F1 │ PR-AUC │ Improvement         │ │
│  │ AWARE Large v4    │ 0.494*   │ 0.484  │ +30.8% vs baseline  │ │
│  │ AWARE Base        │ 0.474    │ 0.486  │ +25.4% vs baseline  │ │
│  │ TF-IDF + LogReg   │ 0.378    │ 0.350  │ (baseline)          │ │
│  │ Random (Prior)    │ 0.083    │ 0.085  │ —                   │ │
│  │ * hover for 95% CI                                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌ Per-Theme Comparison ── [Large v4 │ Base │ Both] ───────────┐ │
│  │ [Interactive grouped bar chart: 8 themes × models]          │ │
│  │ Hover → tooltip with P, R, F1, Support, CI                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌ KEY FINDING ── Multi-Label Bottleneck ──────────────────────┐ │
│  │                                                              │ │
│  │ "When themes appear alone, our model achieves               │ │
│  │  F1 = 0.568-0.896 with PERFECT PRECISION."                 │ │
│  │                                                              │ │
│  │ [Accordion: Single-theme vs Overall F1 comparison chart]     │ │
│  │ [Accordion: What this means for practitioners]               │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌ More Results ── [Threshold │ Overfitting │ Embeddings] ─────┐ │
│  │ [Sub-tabs for different result categories]                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Methodology tab wireframe:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ── Methodology ──                                               │
│                                                                  │
│  ┌ AWARE Architecture (Interactive SVG) ───────────────────────┐ │
│  │                                                              │ │
│  │   [Essay Text] → [DeBERTa-v3 Encoder] → [Mean Pooling]     │ │
│  │                          ↓                                   │ │
│  │              [Position Embedding] → [BiLSTM]                │ │
│  │                          ↓                                   │ │
│  │              [Classification Head] → [8 Theme Predictions]  │ │
│  │                                                              │ │
│  │   Click any component to learn more ▼                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌ 3-Phase Training ──────────────────────────────────────────┐  │
│  │  Phase 1         Phase 2              Phase 3               │  │
│  │  [Frozen]   →   [Progressive]    →   [Head Retrain]        │  │
│  │  8 epochs        40 epochs            5 epochs              │  │
│  │  Head only       Full model + SWA     BiLSTM + Head         │  │
│  │  PR-AUC: 0.398   PR-AUC: 0.521       PR-AUC: 0.522        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌ Loss Function ── [Accordion] ──────────────────────────────┐  │
│  │ ▶ Asymmetric Loss (ASL)                                    │  │
│  │ ▶ R-Drop Consistency                                       │  │
│  │ ▶ Class-Balanced Weights                                   │  │
│  │ ▶ Essay Auxiliary Loss                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌ Regularization Stack ── [Visual layers] ───────────────────┐  │
│  │ 11 techniques visualized as stacked layers                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Tree

```
src/components/
├── layout/
│   ├── AppShell.tsx          # Nav + content area
│   ├── NavBar.tsx            # Top tabs
│   └── PageContainer.tsx     # Consistent padding
├── shared/
│   ├── ThemeBadge.tsx        # Color pill
│   ├── ConfidenceBar.tsx     # Horizontal bar + threshold
│   ├── FilterBar.tsx         # Reusable filter controls
│   ├── LoadingSpinner.tsx
│   └── EmptyState.tsx
├── data-explorer/
│   ├── DataExplorerPage.tsx
│   ├── DataTable.tsx         # TanStack virtual table
│   ├── EssayDetailPanel.tsx
│   ├── EssayRow.tsx
│   └── AnalyzeButton.tsx
├── inference/
│   ├── InferencePage.tsx
│   ├── ModelSelector.tsx
│   ├── SingleEssayTab.tsx
│   ├── BatchUploadTab.tsx
│   ├── InferenceResults.tsx
│   └── ConfidenceBars.tsx
├── exploration/
│   ├── ExplorationPage.tsx
│   ├── SubsetBuilder.tsx
│   ├── ThemeDistributionChart.tsx
│   ├── CoOccurrenceHeatmap.tsx
│   ├── CardinalityChart.tsx
│   ├── TemporalChart.tsx
│   ├── CourseChart.tsx
│   ├── LengthChart.tsx
│   └── EmbeddingScatter.tsx  # Canvas-based
└── paper/
    ├── PaperPage.tsx
    ├── AbstractSection.tsx
    ├── DataSection.tsx
    ├── MethodologySection.tsx
    ├── EngineeringSection.tsx
    ├── ResultsSection.tsx
    ├── ConclusionSection.tsx
    ├── ArchitectureDiagram.tsx  # Interactive SVG
    ├── PipelineVisualization.tsx
    ├── FailureModeCard.tsx
    ├── ComparisonTable.tsx
    └── AnimatedCounter.tsx
```
