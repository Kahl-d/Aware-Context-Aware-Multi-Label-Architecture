import { create } from 'zustand';
import type { InferenceResponse, ModelInfo } from '../types/inference';

interface InferenceState {
  selectedModel: string;
  isProcessing: boolean;
  result: InferenceResponse | null;
  models: ModelInfo[];
  batchProgress: { processed: number; total: number } | null;

  setSelectedModel: (model: string) => void;
  setProcessing: (processing: boolean) => void;
  setResult: (result: InferenceResponse | null) => void;
  setModels: (models: ModelInfo[]) => void;
  setBatchProgress: (progress: { processed: number; total: number } | null) => void;
}

export const useInferenceStore = create<InferenceState>((set) => ({
  selectedModel: 'large_v4',
  isProcessing: false,
  result: null,
  models: [
    {
      id: 'large_v4',
      name: 'AWARE Large v4',
      f1_macro: 0.494,
      params: '360M',
      description: 'DeBERTa-v3-large with 3-phase training and DAPT',
      loaded: false,
    },
    {
      id: 'base',
      name: 'AWARE Base',
      f1_macro: 0.474,
      params: '125M',
      description: 'DeBERTa-v3-base with same architecture',
      loaded: false,
    },
    {
      id: 'tfidf',
      name: 'TF-IDF + LogReg (Baseline)',
      f1_macro: 0.378,
      params: '10K features',
      description: 'Evaluation baseline only — not available for live inference',
      loaded: false,
    },
  ],
  batchProgress: null,

  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setProcessing: (isProcessing) => set({ isProcessing }),
  setResult: (result) => set({ result }),
  setModels: (models) => set({ models }),
  setBatchProgress: (batchProgress) => set({ batchProgress }),
}));
