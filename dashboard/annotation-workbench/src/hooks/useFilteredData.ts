import { useMemo } from 'react';
import { useDataStore } from '../stores/dataStore';
import type { SentenceRecord, EssayAggregate } from '../types/data';
import { THEMES } from '../constants/themes';

export function useFilteredSentences(): { filtered: SentenceRecord[]; total: number } {
  const sentences = useDataStore((s) => s.sentences);
  const filters = useDataStore((s) => s.filters);

  const filtered = useMemo(() => {
    let result = sentences;

    // Annotated status
    if (filters.annotatedStatus === 'annotated') {
      result = result.filter((s) => s.tags.annotated);
    } else if (filters.annotatedStatus === 'unannotated') {
      result = result.filter((s) => !s.tags.annotated);
    }

    // Split filter
    if (filters.split !== 'all') {
      result = result.filter((s) => s.tags.split === filters.split);
    }

    // Theme filter
    if (filters.themes.length > 0) {
      result = result.filter((s) =>
        filters.themes.some((theme) => s.labels[theme as keyof typeof s.labels] === 1)
      );
    }

    // Course
    if (filters.course) {
      result = result.filter((s) => s.course === filters.course);
    }

    // Semester
    if (filters.semester) {
      result = result.filter((s) => s.semester === filters.semester);
    }

    // Year
    if (filters.year) {
      result = result.filter((s) => s.year === filters.year);
    }

    // Prompt
    if (filters.prompt) {
      result = result.filter((s) => s.prompt === filters.prompt);
    }

    // Coder
    if (filters.coder) {
      result = result.filter((s) => s.coder === filters.coder);
    }

    // Search
    if (filters.search) {
      const q = filters.search.toLowerCase();
      result = result.filter(
        (s) =>
          s.sentence.toLowerCase().includes(q) ||
          s.essay_id.toLowerCase().includes(q) ||
          s.alma_id.toLowerCase().includes(q)
      );
    }

    return result;
  }, [sentences, filters]);

  return { filtered, total: sentences.length };
}

/** Aggregate sentences into essay-level records with theme sets */
function aggregateEssays(sentences: SentenceRecord[]): EssayAggregate[] {
  const groups: Record<string, SentenceRecord[]> = {};
  for (const s of sentences) {
    if (!groups[s.essay_id]) groups[s.essay_id] = [];
    groups[s.essay_id].push(s);
  }

  return Object.entries(groups).map(([essay_id, sents]) => {
    sents.sort((a, b) => a.sentence_id - b.sentence_id);
    const first = sents[0];

    // Compute theme SET: each theme counted once if present in ANY sentence
    const themeSet = new Set<string>();
    for (const s of sents) {
      for (const t of THEMES) {
        if (s.labels[t] === 1) themeSet.add(t);
      }
    }

    return {
      essay_id,
      alma_id: first.alma_id,
      course: first.course,
      semester: first.semester,
      year: first.year,
      prompt: first.prompt,
      coder: first.coder,
      sentences: sents,
      sentence_count: sents.length,
      theme_set: [...themeSet].sort(),
      is_class0: themeSet.size === 0,
      annotated: first.tags.annotated,
      split: first.tags.split,
      versions: first.tags.dataset_versions,
      used_for_training: first.tags.used_for_training,
    };
  });
}

/** Essay-level filtering: filters operate on essays, themes checked as essay-level sets */
export function useFilteredEssays(): { filtered: EssayAggregate[]; total: number } {
  const sentences = useDataStore((s) => s.sentences);
  const filters = useDataStore((s) => s.filters);

  const allEssays = useMemo(() => aggregateEssays(sentences), [sentences]);

  const filtered = useMemo(() => {
    let result = allEssays;

    if (filters.annotatedStatus === 'annotated') {
      result = result.filter((e) => e.annotated);
    } else if (filters.annotatedStatus === 'unannotated') {
      result = result.filter((e) => !e.annotated);
    }

    if (filters.split !== 'all') {
      result = result.filter((e) => e.split === filters.split);
    }

    // Theme filter at essay level: essay must have theme in its SET
    if (filters.themes.length > 0) {
      result = result.filter((e) =>
        filters.themes.some((theme) => e.theme_set.includes(theme))
      );
    }

    if (filters.course) {
      result = result.filter((e) => e.course === filters.course);
    }
    if (filters.semester) {
      result = result.filter((e) => e.semester === filters.semester);
    }
    if (filters.year) {
      result = result.filter((e) => e.year === filters.year);
    }
    if (filters.prompt) {
      result = result.filter((e) => e.prompt === filters.prompt);
    }
    if (filters.coder) {
      result = result.filter((e) => e.coder === filters.coder);
    }

    if (filters.search) {
      const q = filters.search.toLowerCase();
      result = result.filter(
        (e) =>
          e.essay_id.toLowerCase().includes(q) ||
          e.alma_id.toLowerCase().includes(q) ||
          e.sentences.some((s) => s.sentence.toLowerCase().includes(q))
      );
    }

    return result;
  }, [allEssays, filters]);

  return { filtered, total: allEssays.length };
}

/** Compute essay-level theme frequency counts: each theme counted once per essay */
export function useEssayThemeCounts(essays: EssayAggregate[]) {
  return useMemo(() => {
    const counts: Record<string, number> = {};
    let class0Count = 0;
    for (const t of THEMES) counts[t] = 0;

    for (const e of essays) {
      if (e.is_class0) {
        class0Count++;
      } else {
        for (const t of e.theme_set) {
          counts[t] = (counts[t] || 0) + 1;
        }
      }
    }

    return { themeCounts: counts, class0Count, totalEssays: essays.length };
  }, [essays]);
}

/** Extract unique values for filter dropdowns from loaded data */
export function useFilterOptions() {
  const sentences = useDataStore((s) => s.sentences);

  return useMemo(() => {
    const courses = new Set<string>();
    const semesters = new Set<string>();
    const years = new Set<string>();
    const prompts = new Set<string>();
    const coders = new Set<string>();

    for (const s of sentences) {
      if (s.course) courses.add(s.course);
      if (s.semester) semesters.add(s.semester);
      if (s.year) years.add(s.year);
      if (s.prompt) prompts.add(s.prompt);
      if (s.coder) coders.add(s.coder);
    }

    return {
      courses: [...courses].sort(),
      semesters: [...semesters].sort(),
      years: [...years].sort(),
      prompts: [...prompts].sort(),
      coders: [...coders].filter(Boolean).sort(),
    };
  }, [sentences]);
}
