import { useState, useRef, useMemo, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { ThemeBadge } from '../shared/ThemeBadge';
import { THEMES, type ThemeName } from '../../constants/themes';
import type { SentenceRecord, EssayAggregate } from '../../types/data';

interface DataTableProps {
  data: SentenceRecord[];
  /** If provided, renders in essay-level view */
  essayData?: EssayAggregate[];
  viewMode?: 'sentence' | 'essay';
  /** Called when user wants to open the detail panel for an essay */
  onEssaySelect?: (essayId: string) => void;
}

function getActiveThemes(labels: SentenceRecord['labels']): ThemeName[] {
  return THEMES.filter((t) => labels[t] === 1);
}

function SplitTag({ split }: { split: string | null }) {
  if (!split) return null;
  const colors: Record<string, string> = {
    train: 'bg-blue-50 text-blue-600 border-blue-100',
    val: 'bg-amber-50 text-amber-600 border-amber-100',
    test: 'bg-emerald-50 text-emerald-600 border-emerald-100',
  };
  return (
    <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded border ${colors[split] || ''}`}>
      {split}
    </span>
  );
}

/** Group sentences by essay_id */
interface EssayGroup {
  essay_id: string;
  sentences: SentenceRecord[];
  course: string;
  semester: string;
  year: string;
  prompt: string;
  coder: string;
  alma_id: string;
  annotated: boolean;
  split: string | null;
  versions: string[];
  themeCount: number;
}

function groupByEssay(data: SentenceRecord[]): EssayGroup[] {
  const groups: Record<string, SentenceRecord[]> = {};
  for (const s of data) {
    if (!groups[s.essay_id]) groups[s.essay_id] = [];
    groups[s.essay_id].push(s);
  }

  return Object.entries(groups).map(([essay_id, sentences]) => {
    sentences.sort((a, b) => a.sentence_id - b.sentence_id);
    const first = sentences[0];
    const allThemes = new Set<string>();
    for (const s of sentences) {
      for (const t of THEMES) {
        if (s.labels[t] === 1) allThemes.add(t);
      }
    }
    return {
      essay_id,
      sentences,
      course: first.course,
      semester: first.semester,
      year: first.year,
      prompt: first.prompt,
      coder: first.coder,
      alma_id: first.alma_id,
      annotated: first.tags.annotated,
      split: first.tags.split,
      versions: first.tags.dataset_versions,
      themeCount: allThemes.size,
    };
  });
}

export function DataTable({ data, essayData, viewMode = 'sentence', onEssaySelect }: DataTableProps) {
  const [expandedEssays, setExpandedEssays] = useState<Set<string>>(new Set());
  const parentRef = useRef<HTMLDivElement>(null);

  const essayGroups = useMemo(() => {
    if (viewMode === 'essay' && essayData) {
      // Convert EssayAggregate[] to EssayGroup[] format for rendering
      return essayData.map((e): EssayGroup => ({
        essay_id: e.essay_id,
        sentences: e.sentences,
        course: e.course,
        semester: e.semester,
        year: e.year,
        prompt: e.prompt,
        coder: e.coder,
        alma_id: e.alma_id,
        annotated: e.annotated,
        split: e.split,
        versions: e.versions,
        themeCount: e.theme_set.length,
      }));
    }
    return groupByEssay(data);
  }, [data, essayData, viewMode]);

  // Build flat list of rows: essay header + expanded sentences
  const rows = useMemo(() => {
    const result: Array<{ type: 'essay'; group: EssayGroup } | { type: 'sentence'; sentence: SentenceRecord; essayId: string }> = [];
    for (const group of essayGroups) {
      result.push({ type: 'essay', group });
      if (expandedEssays.has(group.essay_id)) {
        for (const sent of group.sentences) {
          result.push({ type: 'sentence', sentence: sent, essayId: group.essay_id });
        }
      }
    }
    return result;
  }, [essayGroups, expandedEssays]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => (rows[index].type === 'essay' ? 34 : 32),
    overscan: 15,
  });

  const toggleEssay = useCallback((essayId: string) => {
    setExpandedEssays((prev) => {
      const next = new Set(prev);
      if (next.has(essayId)) next.delete(essayId);
      else next.add(essayId);
      return next;
    });
  }, []);

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center bg-gray-50 border-b border-gray-200 px-3 h-9 text-[11px] font-medium text-gray-500 uppercase tracking-wider select-none">
        <div className="w-7" />
        <div className="w-20 px-1">Essay</div>
        <div className="w-20 px-1">ALMA ID</div>
        <div className="w-18 px-1">Course</div>
        <div className="w-16 px-1">Semester</div>
        <div className="w-14 px-1">Year</div>
        <div className="w-12 px-1 text-center">Sents</div>
        <div className="flex-1 px-1 min-w-0">Text Preview</div>
        <div className="w-40 px-1">{viewMode === 'essay' ? 'Theme Set' : 'Themes'}</div>
        <div className="w-24 px-1">Tags</div>
      </div>

      {/* Virtual scroll body */}
      <div ref={parentRef} className="flex-1 overflow-auto">
        <div style={{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }}>
          {virtualizer.getVirtualItems().map((vItem) => {
            const row = rows[vItem.index];

            if (row.type === 'essay') {
              const g = row.group;
              const isExpanded = expandedEssays.has(g.essay_id);
              const allThemes = new Set<ThemeName>();
              for (const s of g.sentences) {
                for (const t of THEMES) {
                  if (s.labels[t] === 1) allThemes.add(t);
                }
              }

              return (
                <div
                  key={`e-${g.essay_id}`}
                  data-index={vItem.index}
                  ref={virtualizer.measureElement}
                  className={`flex items-center px-3 h-[36px] border-b cursor-pointer transition-all duration-150 ${
                    isExpanded
                      ? 'bg-gray-50 border-gray-300 shadow-sm'
                      : !g.annotated
                      ? 'bg-amber-50/30 border-gray-100 hover:bg-amber-50/60'
                      : 'border-gray-100 hover:bg-gray-50'
                  }`}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    transform: `translateY(${vItem.start}px)`,
                  }}
                  onClick={() => toggleEssay(g.essay_id)}
                >
                  {/* Expand chevron */}
                  <div className="w-7 flex items-center justify-center shrink-0">
                    <svg
                      className={`w-3.5 h-3.5 text-gray-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <div className="w-20 px-1 font-mono text-[12px] text-gray-700 font-medium truncate">
                    {g.essay_id}
                  </div>
                  <div className="w-20 px-1 font-mono text-[11px] text-gray-500 truncate" title={g.alma_id}>
                    {g.alma_id}
                  </div>
                  <div className="w-18 px-1 text-[12px] text-gray-600 truncate">{g.course}</div>
                  <div className="w-16 px-1 text-[12px] text-gray-500 truncate">{g.semester}</div>
                  <div className="w-14 px-1 text-[12px] text-gray-500">{g.year}</div>
                  <div className="w-12 px-1 text-center font-mono text-[12px] text-gray-500">
                    {g.sentences.length}
                  </div>
                  <div className="flex-1 px-1 text-[12px] text-gray-600 truncate min-w-0">
                    {g.sentences.map((s) => s.sentence).join(' ').slice(0, 120)}...
                  </div>
                  <div className="w-40 px-1 flex flex-wrap gap-0.5 shrink-0">
                    {!g.annotated ? (
                      <span className="text-[11px] text-gray-400 italic">unannotated</span>
                    ) : allThemes.size > 0 ? (
                      <>
                        {[...allThemes].map((t) => <ThemeBadge key={t} theme={t} />)}
                        {viewMode === 'essay' && (
                          <span className="text-[10px] font-mono text-gray-400 ml-1">
                            ({allThemes.size})
                          </span>
                        )}
                      </>
                    ) : (
                      <ThemeBadge theme="Class_0" />
                    )}
                  </div>
                  <div className="w-24 px-1 flex items-center gap-1 shrink-0">
                    <SplitTag split={g.split} />
                    {g.sentences[0].tags.used_for_training && (
                      <span className="px-1 py-0.5 text-[9px] font-medium text-indigo-500 bg-indigo-50 border border-indigo-100 rounded">
                        training
                      </span>
                    )}
                    {onEssaySelect && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onEssaySelect(g.essay_id); }}
                        className="w-5 h-5 flex items-center justify-center rounded text-gray-400 hover:text-gray-600 hover:bg-gray-200 transition-colors ml-auto"
                        title="Open detail panel"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
              );
            }

            // Sentence row (expanded)
            const sent = row.sentence;
            const themes = getActiveThemes(sent.labels);

            return (
              <div
                key={`s-${row.essayId}-${sent.sentence_id}`}
                data-index={vItem.index}
                ref={virtualizer.measureElement}
                className="flex items-center px-3 h-[34px] border-b border-gray-50 bg-white border-l-2 border-l-gray-200"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  transform: `translateY(${vItem.start}px)`,
                }}
              >
                <div className="w-7" />
                <div className="w-10 px-1 shrink-0">
                  <span className="text-[11px] font-mono text-gray-400">#{sent.sentence_id}</span>
                </div>
                <div className="flex-1 px-1 truncate text-[12px] text-gray-600 min-w-0">
                  {sent.sentence}
                </div>
                <div className="w-48 px-1 flex flex-wrap gap-0.5 shrink-0">
                  {themes.length > 0 ? (
                    themes.map((t) => <ThemeBadge key={t} theme={t} />)
                  ) : sent.tags.annotated ? (
                    <ThemeBadge theme="Class_0" />
                  ) : (
                    <span className="text-[11px] text-gray-400 italic">no labels</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
