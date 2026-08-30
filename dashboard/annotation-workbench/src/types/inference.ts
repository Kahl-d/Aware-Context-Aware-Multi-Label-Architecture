export interface InferenceRequest {
  text: string;
  model_id: 'large_v4' | 'base' | 'tfidf';
}

export interface SentencePrediction {
  index: number;
  text: string;
  predictions: Record<
    string,
    {
      probability: number;
      predicted: boolean;
      threshold: number;
    }
  >;
}

export interface InferenceResponse {
  model_id: string;
  sentences: SentencePrediction[];
  processing_time_ms: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  f1_macro: number;
  params: string;
  description: string;
  loaded: boolean;
}
