import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import { useDataStore } from '../../stores/dataStore';
import { ThemeBadge } from '../shared/ThemeBadge';
import { LabelEditor } from './LabelEditor';
import { THEMES, type ThemeName } from '../../constants/themes';
import { ROUTES } from '../../constants/routes';
import type { SentenceRecord } from '../../types/data';

interface EssayDetailPanelProps {
  essayId: string;
  onClose: () => void;
}

function getActiveThemes(labels: SentenceRecord['labels']): ThemeName[] {
  return THEMES.filter((t) => labels[t] === 1);
}

export function EssayDetailPanel({ essayId, onClose }: EssayDetailPanelProps) {
  const sentences = useDataStore((s) => s.sentences);
  const annotatorName = useDataStore((s) => s.annotatorName);
  const setAnnotatorName = useDataStore((s) => s.setAnnotatorName);
  const setPrefillInferenceText = useDataStore((s) => s.setPrefillInferenceText);
  const labelEdits = useDataStore((s) => s.labelEdits);
  const navigate = useNavigate();

  const [editingSentence, setEditingSentence] = useState<number | null>(null);
  const [showAnnotatorPrompt, setShowAnnotatorPrompt] = useState(false);
  const [nameInput, setNameInput] = useState(annotatorName);

  const essaySentences = useMemo(
    () =>
      sentences
        .filter((s) => s.essay_id === essayId)
        .sort((a, b) => a.sentence_id - b.sentence_id),
    [sentences, essayId]
  );

  if (essaySentences.length === 0) return null;

  const first = essaySentences[0];
  const isAnnotated = first.tags.annotated;
  const essayEditCount = labelEdits.filter((e) => e.essay_id === essayId).length;

  function handleEditClick(sentenceId: number) {
    if (!annotatorName) {
      setShowAnnotatorPrompt(true);
      setEditingSentence(sentenceId);
      return;
    }
    setEditingSentence(editingSentence === sentenceId ? null : sentenceId);
  }

  function handleSetAnnotator() {
    if (nameInput.trim()) {
      setAnnotatorName(nameInput.trim());
      setShowAnnotatorPrompt(false);
    }
  }

  function handleAnalyzeEssay() {
    const fullText = essaySentences.map((s) => s.sentence).join(' ');
    setPrefillInferenceText(fullText);
    navigate(ROUTES.INFERENCE);
  }

  function handleAnalyzeSentence(sentence: string) {
    setPrefillInferenceText(sentence);
    navigate(ROUTES.INFERENCE);
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed top-0 right-0 bottom-0 z-50 w-full max-w-[480px] bg-white border-l border-gray-200 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-start justify-between px-4 py-3 border-b border-gray-200 bg-gray-50 shrink-0">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 flex-wrap">
              <h3 className="text-[14px] font-semibold text-gray-900 m-0">
                Essay {essayId}
              </h3>
              {first.tags.split && (
                <span
                  className={`px-1.5 py-0.5 text-[10px] font-medium rounded border ${
                    first.tags.split === 'train'
                      ? 'bg-blue-50 text-blue-600 border-blue-100'
                      : first.tags.split === 'val'
                      ? 'bg-amber-50 text-amber-600 border-amber-100'
                      : 'bg-emerald-50 text-emerald-600 border-emerald-100'
                  }`}
                >
                  {first.tags.split}
                </span>
              )}
              {!isAnnotated && (
                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded border bg-orange-50 text-orange-600 border-orange-100">
                  unannotated
                </span>
              )}
              {essayEditCount > 0 && (
                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded border bg-violet-50 text-violet-600 border-violet-100">
                  {essayEditCount} edit{essayEditCount !== 1 ? 's' : ''}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1 text-[11px] text-gray-500 flex-wrap">
              <span>{first.course}</span>
              <span className="text-gray-300">|</span>
              <span>{first.semester} {first.year}</span>
              {first.coder && (
                <>
                  <span className="text-gray-300">|</span>
                  <span>{first.coder}</span>
                </>
              )}
            </div>
            {first.prompt && (
              <p className="mt-1 text-[11px] text-gray-400 italic leading-snug m-0 truncate" title={first.prompt}>
                &ldquo;{first.prompt}&rdquo;
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors shrink-0 ml-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Stats + Analyze button */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100 bg-white shrink-0">
          <div className="flex items-center gap-3 text-[11px] text-gray-500">
            <span>
              <span className="font-mono font-medium text-gray-700">{essaySentences.length}</span> sentences
            </span>
            <span className="font-mono text-gray-400" title={first.alma_id}>
              {first.alma_id}
            </span>
          </div>
          <button
            onClick={handleAnalyzeEssay}
            className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-gray-600 bg-white border border-gray-200 rounded hover:bg-gray-50 hover:border-gray-300 transition-colors"
            title="Analyze full essay with AWARE model"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Analyze
          </button>
        </div>

        {/* Annotator prompt */}
        {showAnnotatorPrompt && (
          <div className="px-4 py-2.5 border-b border-gray-100 bg-amber-50/50 shrink-0">
            <p className="text-[11px] text-gray-600 mb-1.5 m-0">
              Enter your name to start annotating:
            </p>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                placeholder="Your name..."
                autoFocus
                onKeyDown={(e) => e.key === 'Enter' && handleSetAnnotator()}
                className="flex-1 h-7 px-2 text-[12px] border border-gray-200 rounded bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-gray-300"
              />
              <button
                onClick={handleSetAnnotator}
                disabled={!nameInput.trim()}
                className="px-2.5 h-7 text-[11px] font-medium text-white bg-gray-800 rounded hover:bg-gray-700 disabled:opacity-40 transition-colors"
              >
                Set
              </button>
              <button
                onClick={() => { setShowAnnotatorPrompt(false); setEditingSentence(null); }}
                className="text-[11px] text-gray-500 hover:text-gray-700 px-1"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Sentences */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="px-4 py-3 space-y-3">
            {essaySentences.map((sent) => {
              const themes = getActiveThemes(sent.labels);
              const isEditing = editingSentence === sent.sentence_id;
              const hasEdits = labelEdits.some(
                (e) => e.essay_id === essayId && e.sentence_id === sent.sentence_id
              );

              return (
                <div key={sent.sentence_id} className="group">
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] font-mono text-gray-400 mt-1 w-4 text-right shrink-0">
                      {sent.sentence_id}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] text-gray-700 leading-relaxed m-0 break-words">
                        {sent.sentence}
                      </p>
                      <div className="flex items-center gap-1 mt-1">
                        <div className="flex flex-wrap gap-1 flex-1 min-w-0">
                          {!isAnnotated && themes.length === 0 ? (
                            <span className="text-[10px] text-gray-400 italic">no labels</span>
                          ) : themes.length > 0 ? (
                            themes.map((t) => <ThemeBadge key={t} theme={t} size="md" showFull />)
                          ) : (
                            <ThemeBadge theme="Class_0" size="md" showFull />
                          )}
                          {hasEdits && (
                            <span className="text-[9px] text-violet-500 font-medium">edited</span>
                          )}
                        </div>

                        {/* Action buttons */}
                        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                          <button
                            onClick={() => handleEditClick(sent.sentence_id)}
                            className="w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                            title="Edit labels"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                            </svg>
                          </button>
                          <button
                            onClick={() => handleAnalyzeSentence(sent.sentence)}
                            className="w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                            title="Analyze with model"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                          </button>
                        </div>
                      </div>

                      {/* Inline label editor */}
                      {isEditing && annotatorName && (
                        <LabelEditor
                          sentence={sent}
                          onDone={() => setEditingSentence(null)}
                        />
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer: export edits */}
        {labelEdits.length > 0 && (
          <div className="px-4 py-2 border-t border-gray-200 bg-gray-50 flex items-center justify-between shrink-0">
            <span className="text-[10px] text-gray-500">
              <span className="font-mono font-medium text-gray-700">{labelEdits.length}</span> label edits this session
            </span>
            <button
              onClick={() => {
                const json = JSON.stringify(labelEdits, null, 2);
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `label_edits_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="px-2 py-0.5 text-[10px] font-medium text-gray-600 bg-white border border-gray-200 rounded hover:bg-gray-50 transition-colors"
            >
              Export edits
            </button>
          </div>
        )}
      </div>
    </>
  );
}
