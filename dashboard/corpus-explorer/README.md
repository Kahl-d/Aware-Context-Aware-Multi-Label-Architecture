# AWARE Dashboard Documentation

## Overview

Research dashboard for exploring and analyzing Cultural Capital theme annotations in student essays.

## Quick Start

```bash
cd dashboard
npm install
npm run dev
# Opens at http://localhost:5173
```

## Data Sources

Dashboard reads from `public/` folder:
- `master_data.json` - All essays with sentence-level annotations
- `stats.json` - Aggregated statistics

## Tabs

### Data (Default)
Spreadsheet-like table of all essays with:
- **Columns**: ID, Year, Course, Sentences, Themes, Essay Text
- **Features**: Search, filter, sort, pagination
- **Expand**: Click row for full essay + annotated sentences
- **Legend**: Toggle theme color reference

### Themes
- Theme distribution table (sentence-level counts)
- Class imbalance warning

### Overview
- Stats cards (essays, sentences, themes, courses)
- Bar charts by year and course

## Theme Colors

| Theme | Color |
|-------|-------|
| class_0 | Gray |
| Aspirational | Green |
| Attainment | Blue |
| Navigational | Purple |
| Perseverance | Amber |
| Resistance | Red |
| Familial | Cyan |
| Social | Pink |
| Spiritual | Indigo |
| First Gen | Teal |
| Filial Piety | Orange |
| Community Consciousness | Lime |

## Updating Data

After processing new essays:

```bash
# 1. Rebuild master data
cd data/scripts
python3 build_master_data.py --output-json

# 2. Regenerate stats
python3 generate_stats.py

# 3. Copy to dashboard
cp ../processed/master_data.json ../../dashboard/public/
```

## Tech Stack
- React 18 + Vite
- No external chart libraries
- CSS-based visualizations
