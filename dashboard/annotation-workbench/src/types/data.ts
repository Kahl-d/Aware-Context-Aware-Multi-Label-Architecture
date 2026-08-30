import type { ThemeName } from '../constants/themes';

export interface SentenceRecord {
  essay_id: string;
  sentence_id: number;
  sentence: string;
  sentence_length: number;
  alma_id: string;
  course: string;
  semester: string;
  year: string;
  prompt: string;
  source_file: string;
  coder: string;

  labels: Record<ThemeName | 'Class_0', number | null>;

  tags: {
    annotated: boolean;
    dataset_versions: string[];
    used_for_training: boolean;
    split: 'train' | 'val' | 'test' | null;
    dropped_reason: string | null;
  };

  predictions: {
    model: string | null;
    themes: Record<string, number> | null;
    binary: Record<string, 0 | 1> | null;
    timestamp: string | null;
  } | null;
}

export interface EssayRecord {
  essay_id: string;
  alma_id: string;
  course: string;
  semester: string;
  year: string;
  prompt: string;
  coder: string;
  sentence_count: number;
  annotated_count: number;
  sentence_ids: number[];
  tags: {
    annotated: boolean;
    used_for_training: boolean;
    split: 'train' | 'val' | 'test' | null;
    dataset_versions: string[];
  };
}

export interface DatasetVersionSummary {
  version: string;
  sentences: number;
  essays: number;
  themes: number;
  description: string;
}

export interface FilterState {
  viewMode: 'sentence' | 'essay';
  datasetVersion: string;
  annotatedStatus: 'all' | 'annotated' | 'unannotated';
  split: 'all' | 'train' | 'val' | 'test';
  themes: string[];
  course: string;
  semester: string;
  year: string;
  prompt: string;
  coder: string;
  search: string;
}

/** Essay-level aggregation: each theme counted once per essay */
export interface EssayAggregate {
  essay_id: string;
  alma_id: string;
  course: string;
  semester: string;
  year: string;
  prompt: string;
  coder: string;
  sentences: SentenceRecord[];
  sentence_count: number;
  /** Set of themes present in this essay (each counted once) */
  theme_set: string[];
  /** True only if ALL sentences are Class_0 (no themes anywhere) */
  is_class0: boolean;
  annotated: boolean;
  split: string | null;
  versions: string[];
  used_for_training: boolean;
}
