import Papa from 'papaparse';
import type { SentenceRecord } from '../types/data';
import { THEMES } from '../constants/themes';

export function exportSentencesCSV(sentences: SentenceRecord[], filename = 'alma_export.csv') {
  const rows = sentences.map((s) => ({
    essay_id: s.essay_id,
    sentence_id: s.sentence_id,
    sentence: s.sentence,
    sentence_length: s.sentence_length,
    alma_id: s.alma_id,
    course: s.course,
    semester: s.semester,
    year: s.year,
    prompt: s.prompt,
    coder: s.coder,
    ...Object.fromEntries(THEMES.map((t) => [t, s.labels[t] ?? ''])),
    Class_0: s.labels.Class_0 ?? '',
    annotated: s.tags.annotated ? 1 : 0,
    used_for_training: s.tags.used_for_training ? 1 : 0,
    split: s.tags.split ?? '',
    dataset_versions: s.tags.dataset_versions.join(';'),
  }));

  const csv = Papa.unparse(rows);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
