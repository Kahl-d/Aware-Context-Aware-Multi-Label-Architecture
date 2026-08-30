import { useState, useCallback, useRef, useEffect } from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { ConfidenceBar } from '../components/shared/ConfidenceBar';
import { ThemeBadge } from '../components/shared/ThemeBadge';
import { THEMES, THEME_COLORS, type ThemeName } from '../constants/themes';
import { useInferenceStore } from '../stores/inferenceStore';
import { useDataStore } from '../stores/dataStore';
import { inferSingle, inferBatch } from '../lib/api';
import type { SentencePrediction } from '../types/inference';

function ModelSelector() {
  const { selectedModel, setSelectedModel, models } = useInferenceStore();
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[12px] font-medium text-gray-500 uppercase tracking-wider mb-3 m-0">
        Select Model
      </h3>
      <div className="space-y-2">
        {models.map((m) => {
          const disabled = m.id === 'tfidf';
          return (
            <label
              key={m.id}
              className={`flex items-center gap-3 px-3 py-2 rounded-md border transition-colors ${
                disabled
                  ? 'border-gray-100 opacity-50 cursor-not-allowed'
                  : selectedModel === m.id
                  ? 'border-gray-800 bg-gray-50 cursor-pointer'
                  : 'border-gray-100 hover:border-gray-200 cursor-pointer'
              }`}
            >
              <input
                type="radio"
                name="model"
                value={m.id}
                checked={selectedModel === m.id}
                onChange={() => !disabled && setSelectedModel(m.id)}
                disabled={disabled}
                className="accent-gray-800"
              />
              <div className="flex-1">
                <span className="text-[13px] font-medium text-gray-800">{m.name}</span>
                <span className="text-[12px] text-gray-400 ml-2">F1={m.f1_macro} | {m.params}</span>
                {disabled && <span className="text-[11px] text-gray-400 block">Evaluation baseline only</span>}
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}

/** Get the dominant predicted theme color for a sentence */
function getDominantColor(sent: SentencePrediction): string {
  let maxProb = 0;
  let color = 'transparent';
  for (const theme of THEMES) {
    const p = sent.predictions[theme];
    if (p && p.predicted && p.probability > maxProb) {
      maxProb = p.probability;
      color = THEME_COLORS[theme];
    }
  }
  return color;
}

/** Get opacity based on max predicted probability */
function getIntensity(sent: SentencePrediction): number {
  let maxProb = 0;
  for (const theme of THEMES) {
    const p = sent.predictions[theme];
    if (p && p.predicted && p.probability > maxProb) {
      maxProb = p.probability;
    }
  }
  return Math.max(0.08, maxProb * 0.35);
}

function HighlightedEssay({ results }: { results: SentencePrediction[] }) {
  const [activeSent, setActiveSent] = useState<number | null>(null);

  return (
    <div className="space-y-6">
      {/* Highlighted essay text */}
      <div>
        <h3 className="text-[12px] font-medium text-gray-500 uppercase tracking-wider mb-2 m-0">
          Analyzed Essay
        </h3>
        <div className="border border-gray-200 rounded-lg p-5 bg-white leading-[1.8] text-[15px] text-gray-800">
          {results.map((sent, i) => {
            const color = getDominantColor(sent);
            const intensity = getIntensity(sent);
            const predicted = THEMES.filter((t) => sent.predictions[t]?.predicted);
            const isActive = activeSent === i;

            return (
              <span
                key={i}
                className={`cursor-pointer rounded-sm transition-all duration-200 ${isActive ? 'ring-2 ring-gray-300' : ''}`}
                style={{
                  backgroundColor: predicted.length > 0 ? `${color}${Math.round(intensity * 255).toString(16).padStart(2, '0')}` : 'transparent',
                  borderBottom: predicted.length > 0 ? `2px solid ${color}40` : 'none',
                  padding: '1px 2px',
                }}
                onClick={() => setActiveSent(isActive ? null : i)}
                title={predicted.length > 0 ? predicted.join(', ') : 'No themes detected'}
              >
                {sent.text}{' '}
              </span>
            );
          })}
        </div>
        <p className="text-[11px] text-gray-400 mt-2 m-0">
          Click any sentence to see detailed predictions below. Color intensity reflects prediction confidence.
        </p>
      </div>

      {/* Per-sentence detail */}
      <div>
        <h3 className="text-[12px] font-medium text-gray-500 uppercase tracking-wider mb-2 m-0">
          Sentence-Level Predictions
        </h3>
        <div className="space-y-2">
          {results.map((sent) => {
            const predicted = THEMES.filter((t) => sent.predictions[t]?.predicted);
            const isActive = activeSent === sent.index;

            return (
              <div
                key={sent.index}
                className={`border rounded-lg p-3 bg-white transition-all duration-200 ${
                  isActive ? 'border-gray-300 shadow-sm' : 'border-gray-100'
                }`}
                onClick={() => setActiveSent(isActive ? null : sent.index)}
              >
                <div className="flex items-start gap-2 mb-1.5">
                  <span className="text-[11px] font-mono text-gray-400 mt-0.5 shrink-0 w-5 text-right">
                    {sent.index + 1}
                  </span>
                  <p className="text-[14px] text-gray-700 m-0 leading-relaxed flex-1">
                    {sent.text}
                  </p>
                </div>
                {predicted.length > 0 && (
                  <div className="flex gap-1 mb-2 ml-7">
                    {predicted.map((t) => (
                      <ThemeBadge key={t} theme={t as ThemeName} size="md" showFull />
                    ))}
                  </div>
                )}
                {(isActive || predicted.length > 0) && (
                  <div className="space-y-0.5 ml-7">
                    {THEMES.map((theme) => {
                      const p = sent.predictions[theme];
                      if (!p) return null;
                      return (
                        <ConfidenceBar
                          key={theme}
                          theme={theme}
                          probability={p.probability}
                          threshold={p.threshold}
                          predicted={p.predicted}
                        />
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function SingleEssayTab() {
  const [text, setText] = useState('');
  const { selectedModel, isProcessing, setProcessing } = useInferenceStore();
  const prefillText = useDataStore((s) => s.prefillInferenceText);
  const setPrefillInferenceText = useDataStore((s) => s.setPrefillInferenceText);
  const [results, setResults] = useState<SentencePrediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [timeMs, setTimeMs] = useState(0);
  const editorRef = useRef<HTMLDivElement>(null);

  // Consume prefill text from Data Explorer
  useEffect(() => {
    if (prefillText) {
      setText(prefillText);
      if (editorRef.current) {
        editorRef.current.textContent = prefillText;
      }
      setPrefillInferenceText(null); // Clear so it doesn't re-trigger
    }
  }, [prefillText, setPrefillInferenceText]);

  const sentenceCount = text.trim()
    ? text.trim().split(/(?<=[.!?])\s+/).filter((s) => s.length > 3).length
    : 0;

  const handleAnalyze = useCallback(async () => {
    if (!text.trim()) return;
    setProcessing(true);
    setError(null);
    try {
      const res = await inferSingle(text, selectedModel);
      setResults(res.sentences);
      setTimeMs(res.processing_time_ms);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
    } finally {
      setProcessing(false);
    }
  }, [text, selectedModel, setProcessing]);

  return (
    <div className="space-y-4">
      {/* Input area — clean, open feel */}
      {!results && (
        <div>
          <div
            ref={editorRef}
            contentEditable
            suppressContentEditableWarning
            onInput={(e) => setText(e.currentTarget.textContent || '')}
            data-placeholder="Paste an essay or sentence here..."
            className="min-h-[160px] px-4 py-3 text-[15px] leading-relaxed border border-gray-200 rounded-lg bg-white text-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-300 empty:before:content-[attr(data-placeholder)] empty:before:text-gray-400"
          />
          <div className="flex items-center justify-between mt-3">
            <span className="text-[12px] text-gray-400">
              {sentenceCount} sentence{sentenceCount !== 1 ? 's' : ''} detected
            </span>
            <button
              onClick={handleAnalyze}
              disabled={isProcessing || !text.trim()}
              className="px-5 h-9 text-[13px] font-medium text-white bg-gray-800 rounded-lg hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {isProcessing ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => { setResults(null); setText(''); }}
                className="text-[12px] text-gray-500 hover:text-gray-700 underline"
              >
                &larr; New analysis
              </button>
              <span className="text-[11px] text-gray-400">
                {results.length} sentences | {timeMs}ms
              </span>
            </div>
          </div>
          <HighlightedEssay results={results} />
        </div>
      )}

      {error && (
        <div className="px-3 py-2 text-[13px] text-red-600 bg-red-50 border border-red-100 rounded-md">
          {error}
        </div>
      )}
    </div>
  );
}

function BatchUploadTab() {
  const { selectedModel, isProcessing, setProcessing } = useInferenceStore();
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = useCallback(async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setProcessing(true);
    setError(null);
    setDone(false);
    try {
      const res = await inferBatch(file, selectedModel);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'aware_predictions.csv';
      link.click();
      URL.revokeObjectURL(url);
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch inference failed');
    } finally {
      setProcessing(false);
    }
  }, [selectedModel, setProcessing]);

  return (
    <div className="space-y-4">
      <div className="border border-gray-200 rounded-lg p-4 bg-white">
        <p className="text-[13px] text-gray-600 mb-2 m-0">
          Upload a CSV file with your essays. The model will analyze each essay and return per-sentence CCW theme predictions.
        </p>
        <p className="text-[12px] text-gray-500 mb-3 m-0">
          Expected columns: <code className="text-[11px] bg-gray-100 px-1 py-0.5 rounded">essay_id</code> and{' '}
          <code className="text-[11px] bg-gray-100 px-1 py-0.5 rounded">essay_text</code>
        </p>
        <div className="flex items-center gap-3">
          <input ref={fileRef} type="file" accept=".csv" className="text-[13px] text-gray-600" />
          <button
            onClick={handleUpload}
            disabled={isProcessing}
            className="px-4 h-8 text-[13px] font-medium text-white bg-gray-800 rounded-md hover:bg-gray-700 disabled:opacity-40 transition-colors"
          >
            {isProcessing ? 'Processing...' : 'Upload & Analyze'}
          </button>
        </div>
      </div>
      {error && (
        <div className="px-3 py-2 text-[13px] text-red-600 bg-red-50 border border-red-100 rounded-md">{error}</div>
      )}
      {done && (
        <div className="px-3 py-2 text-[13px] text-green-700 bg-green-50 border border-green-100 rounded-md">
          Results CSV downloaded successfully.
        </div>
      )}
    </div>
  );
}

export function InferencePage() {
  const [tab, setTab] = useState<'single' | 'batch'>('single');

  return (
    <PageContainer>
      <div className="mb-4">
        <h1 className="text-[15px] font-semibold text-gray-900 m-0">Model Inference</h1>
        <p className="text-[12px] text-gray-500 mt-0.5">
          Analyze essays using trained AWARE models — single essay or batch CSV
        </p>
      </div>

      <div className="grid grid-cols-[280px_1fr] gap-6">
        <ModelSelector />

        <div className="space-y-4">
          <div className="flex gap-1 border-b border-gray-200">
            {(['single', 'batch'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2 text-[13px] font-medium border-b-2 transition-colors ${
                  tab === t
                    ? 'border-gray-800 text-gray-800'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {t === 'single' ? 'Single Essay' : 'Batch CSV'}
              </button>
            ))}
          </div>
          {tab === 'single' ? <SingleEssayTab /> : <BatchUploadTab />}
        </div>
      </div>
    </PageContainer>
  );
}
