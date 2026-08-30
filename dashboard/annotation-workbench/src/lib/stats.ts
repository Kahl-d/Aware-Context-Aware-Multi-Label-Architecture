import type { SentenceRecord } from '../types/data';
import { THEMES } from '../constants/themes';

/** Count how many sentences have each theme active */
export function themeDistribution(sentences: SentenceRecord[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const t of THEMES) counts[t] = 0;
  counts['Class_0'] = 0;

  for (const s of sentences) {
    if (!s.tags.annotated) continue;
    let hasTheme = false;
    for (const t of THEMES) {
      if (s.labels[t] === 1) {
        counts[t]++;
        hasTheme = true;
      }
    }
    if (!hasTheme) counts['Class_0']++;
  }
  return counts;
}

/** 8x8 co-occurrence matrix */
export function coOccurrenceMatrix(
  sentences: SentenceRecord[]
): { themes: string[]; matrix: number[][] } {
  const n = THEMES.length;
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  for (const s of sentences) {
    if (!s.tags.annotated) continue;
    const active: number[] = [];
    for (let i = 0; i < n; i++) {
      if (s.labels[THEMES[i]] === 1) active.push(i);
    }
    for (const i of active) {
      for (const j of active) {
        matrix[i][j]++;
      }
    }
  }
  return { themes: [...THEMES], matrix };
}

/** Label cardinality distribution: 0, 1, 2, 3, 4+ themes per sentence */
export function cardinalityDistribution(sentences: SentenceRecord[]): Record<string, number> {
  const dist: Record<string, number> = { '0': 0, '1': 0, '2': 0, '3': 0, '4+': 0 };

  for (const s of sentences) {
    if (!s.tags.annotated) continue;
    let count = 0;
    for (const t of THEMES) {
      if (s.labels[t] === 1) count++;
    }
    if (count === 0) dist['0']++;
    else if (count === 1) dist['1']++;
    else if (count === 2) dist['2']++;
    else if (count === 3) dist['3']++;
    else dist['4+']++;
  }
  return dist;
}

/** Theme counts by a grouping dimension (course, year, semester) */
export function themesByGroup(
  sentences: SentenceRecord[],
  groupBy: 'course' | 'year' | 'semester'
): { group: string; [key: string]: string | number }[] {
  const groups: Record<string, Record<string, number>> = {};

  for (const s of sentences) {
    if (!s.tags.annotated) continue;
    const key = s[groupBy] || 'Unknown';
    if (!groups[key]) {
      groups[key] = {};
      for (const t of THEMES) groups[key][t] = 0;
    }
    for (const t of THEMES) {
      if (s.labels[t] === 1) groups[key][t]++;
    }
  }

  return Object.entries(groups)
    .map(([group, counts]) => ({ group, ...counts }))
    .sort((a, b) => String(a.group).localeCompare(String(b.group)));
}

/** Sentence length distribution in buckets */
export function lengthDistribution(
  sentences: SentenceRecord[]
): { bucket: string; count: number }[] {
  const buckets: Record<string, number> = {
    '1-5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-30': 0, '31-50': 0, '50+': 0,
  };

  for (const s of sentences) {
    const len = s.sentence_length;
    if (len <= 5) buckets['1-5']++;
    else if (len <= 10) buckets['6-10']++;
    else if (len <= 15) buckets['11-15']++;
    else if (len <= 20) buckets['16-20']++;
    else if (len <= 30) buckets['21-30']++;
    else if (len <= 50) buckets['31-50']++;
    else buckets['50+']++;
  }

  return Object.entries(buckets).map(([bucket, count]) => ({ bucket, count }));
}
