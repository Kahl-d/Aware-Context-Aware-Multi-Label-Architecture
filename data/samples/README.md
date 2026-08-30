# Sample Data

> **The full datasets are intentionally NOT in this repository.** They contain
> the complete corpus of student reflective essays and are large. This folder
> holds a **small, representative, pseudonymized sample** so you can see the
> exact schema the pipeline and dashboard use. To get the full data, run the
> pipeline (`../data-pipeline/`) on the ALMA source files, or ask the ALMA
> research team / Khalid for the generated CSVs and JSONs.

## Files

| File | Records | What it is |
|---|---|---|
| `essays.sample.json` | 2 | Essay-level records (one `Why am I here?`, one `What do I do when life gets challenging?`). |
| `sentences.sample.json` | 18 | The sentence-level records for those 2 essays — labels, tags, predictions. |
| `dataset_versions.json` | 4 | Version metadata (V1→V4 sentence/essay/theme counts). Full file, no student text. |
| `theme_colors.json` | — | Hex colors per theme used by the dashboard. |

## Privacy note

Students are already pseudonymized. There are **no names** — the only identifier
is `alma_id` (e.g. `F20.PHYS0122.01.024.075` = semester · course · section ·
anonymous IDs). The two sample essays were eyeballed for embedded personal
details before inclusion. If you add more samples, **review the sentence text
for names/PII first.**

---

## Sentence record schema (`sentences.json`)

```jsonc
{
  "essay_id": "852",            // string; groups sentences into an essay
  "sentence_id": 1,             // int; 1-based within the essay
  "sentence": "The reason why I am here (PHYS 122) is ...",
  "sentence_length": 96,        // character count
  "alma_id": "F20.PHYS0122.01.024.075",  // pseudonymized student/essay id
  "course": "PHYS122",
  "semester": "Fall",
  "year": "2020",
  "prompt": "Why am I here?",   // the reflective-writing prompt
  "source_file": "merged_reconciled_annotations_complete.xlsx",
  "coder": "reconciled",        // who/how it was annotated (merged / reconciled / a coder id)
  "labels": {                   // multi-label ground truth (0/1 per theme)
    "Attainment": 0, "Aspirational": 1, "Navigational": 0, "Resistance": 0,
    "Perseverance": 0, "Social": 0, "Spiritual": 0, "Familial_Capital": 0,
    "Class_0": 0                // Class_0 = 1 means "no CCW theme present"
  },
  "tags": {
    "annotated": true,          // false = unannotated essay (no human labels)
    "dataset_versions": ["v1","v2","v3","v4"],  // which versions keep this sentence
    "used_for_training": true,  // present in the final V4 training set
    "split": "train",           // "train" | "val" | "test" | null
    "dropped_reason": null      // null | "semantic_cleaning" | "boundary" | "theme_consolidation"
  },
  "predictions": null           // model output filled in at inference time; null until then
}
```

## Essay record schema (`essays.json`)

```jsonc
{
  "essay_id": "97",
  "alma_id": "S25.PHYS102.109595",
  "course": "PHYS102",
  "semester": "Spring",
  "year": "2025",
  "prompt": "What do I do when life gets challenging?",
  "coder": "merged",
  "sentence_count": 9,          // total sentences in the essay
  "annotated_count": 9,         // how many were human-annotated
  "sentence_ids": [1,2,3,4,5,6,7,9,10],  // ids into sentences.json
  "tags": {
    "annotated": true,
    "used_for_training": true,
    "split": "test",
    "dataset_versions": ["v1","v2","v3","v4"]
  }
}
```

## The 8 CCW themes + Class_0

`Attainment` · `Aspirational` · `Navigational` · `Resistance` · `Perseverance` ·
`Social` · `Spiritual` · `Familial_Capital`, plus `Class_0` (no theme present).
A sentence can carry **multiple** theme labels at once (multi-label). See
`../docs/` for what each theme means.
