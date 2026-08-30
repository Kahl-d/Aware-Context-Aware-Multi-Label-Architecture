# `public/data/` — dashboard data files

The frontend fetches two big files at runtime that are **NOT committed** (they
hold the full student corpus and are large):

- `sentences.json`  (~19 MB)
- `essays.json`     (~1.4 MB)

Both are listed in `.gitignore`. The smaller files here (`evaluation_results.json`,
`group_analysis.json`, `splits_stats.json`, `training_history.json`,
`dataset_versions.json`, `theme_colors.json`) are model results / metadata and
**are** committed.

## Quick start (demo data — boots immediately)

Two tracked demo files ship in this folder. Copy them into the names the app
expects, then start the dev server:

```bash
cp essays.demo.json    essays.json
cp sentences.demo.json sentences.json
# then, from the dashboard/ root:  npm install && npm run dev
```

The app will load with **2 sample essays** so you can see every page working.

## Full data

To run against the complete dataset, regenerate the JSONs from the processed
CSVs with the prep script and drop them in here:

```bash
python ../../data-prep/prepare_all.py   # update the paths inside first
```

(Or obtain `essays.json` / `sentences.json` from the ALMA team / Khalid.)

> Because `essays.json` and `sentences.json` are gitignored, you will not
> accidentally commit the full student corpus when you generate it here.
