import { create } from 'zustand';
import type { SentenceRecord, EssayRecord, FilterState } from '../types/data';
import type { ThemeName } from '../constants/themes';

/** A single label edit record */
export interface LabelEdit {
  essay_id: string;
  sentence_id: number;
  theme: ThemeName | 'Class_0';
  old_value: number | null;
  new_value: number;
  annotator: string;
  timestamp: string; // ISO
}

interface DataState {
  sentences: SentenceRecord[];
  essays: EssayRecord[];
  isLoading: boolean;
  error: string | null;
  filters: FilterState;

  // Annotation editing
  annotatorName: string;
  labelEdits: LabelEdit[];
  /** Text to prefill on the Inference page (set from Data Explorer) */
  prefillInferenceText: string | null;

  setSentences: (sentences: SentenceRecord[]) => void;
  setEssays: (essays: EssayRecord[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setFilter: <K extends keyof FilterState>(key: K, value: FilterState[K]) => void;
  resetFilters: () => void;
  setAnnotatorName: (name: string) => void;
  /** Update a label in-place and record the edit */
  updateLabel: (essayId: string, sentenceId: number, theme: ThemeName | 'Class_0', value: number, annotator: string) => void;
  setPrefillInferenceText: (text: string | null) => void;
}

const defaultFilters: FilterState = {
  viewMode: 'sentence',
  datasetVersion: 'all',
  annotatedStatus: 'all',
  split: 'all',
  themes: [],
  course: '',
  semester: '',
  year: '',
  prompt: '',
  coder: '',
  search: '',
};

export const useDataStore = create<DataState>((set, get) => ({
  sentences: [],
  essays: [],
  isLoading: true,
  error: null,
  filters: { ...defaultFilters },
  annotatorName: '',
  labelEdits: [],
  prefillInferenceText: null,

  setSentences: (sentences) => set({ sentences }),
  setEssays: (essays) => set({ essays }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  setFilter: (key, value) =>
    set((state) => ({
      filters: { ...state.filters, [key]: value },
    })),
  resetFilters: () => set({ filters: { ...defaultFilters } }),
  setAnnotatorName: (annotatorName) => set({ annotatorName }),

  updateLabel: (essayId, sentenceId, theme, value, annotator) => {
    const { sentences, labelEdits } = get();
    const idx = sentences.findIndex(
      (s) => s.essay_id === essayId && s.sentence_id === sentenceId
    );
    if (idx === -1) return;

    const sent = sentences[idx];
    const oldValue = sent.labels[theme as keyof typeof sent.labels];

    // Update sentence labels in-place (new array for React reactivity)
    const updated = [...sentences];
    updated[idx] = {
      ...sent,
      labels: { ...sent.labels, [theme]: value },
    };

    const edit: LabelEdit = {
      essay_id: essayId,
      sentence_id: sentenceId,
      theme: theme as ThemeName | 'Class_0',
      old_value: oldValue,
      new_value: value,
      annotator,
      timestamp: new Date().toISOString(),
    };

    set({ sentences: updated, labelEdits: [...labelEdits, edit] });
  },

  setPrefillInferenceText: (prefillInferenceText) => set({ prefillInferenceText }),
}));
