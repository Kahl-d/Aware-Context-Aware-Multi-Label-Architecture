import { useCallback, useState } from 'react';
import { FilterBar } from '../components/data-explorer/FilterBar';
import { DataTable } from '../components/data-explorer/DataTable';
import { EssayDetailPanel } from '../components/data-explorer/EssayDetailPanel';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { useDataLoader } from '../hooks/useDataLoader';
import { useDataStore } from '../stores/dataStore';
import { useFilteredSentences, useFilteredEssays, useEssayThemeCounts } from '../hooks/useFilteredData';
import { exportSentencesCSV } from '../lib/csv';

export function DataExplorerPage() {
  const { isLoading, error } = useDataLoader();
  const [detailEssayId, setDetailEssayId] = useState<string | null>(null);
  const viewMode = useDataStore((s) => s.filters.viewMode);

  const { filtered: filteredSentences, total: totalSentences } = useFilteredSentences();
  const { filtered: filteredEssays, total: totalEssays } = useFilteredEssays();
  const essayThemeCounts = useEssayThemeCounts(filteredEssays);

  const isEssayView = viewMode === 'essay';
  const filteredCount = isEssayView ? filteredEssays.length : filteredSentences.length;
  const totalCount = isEssayView ? totalEssays : totalSentences;

  const handleExport = useCallback(() => {
    if (isEssayView) {
      // Export all sentences from filtered essays
      const allSentences = filteredEssays.flatMap((e) => e.sentences);
      exportSentencesCSV(allSentences, 'alma_essays_export.csv');
    } else {
      exportSentencesCSV(filteredSentences);
    }
  }, [isEssayView, filteredEssays, filteredSentences]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-56px)]">
        <LoadingSpinner message="Loading 27,186 sentences..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-56px)] text-red-500 text-sm">
        Failed to load data: {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] overflow-hidden max-w-[1400px] mx-auto w-full">
      {/* Header */}
      <div className="px-6 pt-4 pb-2 shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-[15px] font-semibold text-gray-900 m-0">Data Explorer</h1>
            <p className="text-[12px] text-gray-500 mt-0.5">
              Browse all ALMA essays — click any row to expand, or use the arrow to open detail panel with annotation editing
            </p>
          </div>
          <span className="text-[11px] font-mono text-gray-400">
            {isEssayView ? 'Essay View' : 'Sentence View'} | {filteredCount.toLocaleString()} items
          </span>
        </div>
      </div>

      {/* Filters */}
      <div className="px-6 pb-3 shrink-0">
        <FilterBar
          filteredCount={filteredCount}
          totalCount={totalCount}
          onExport={handleExport}
          essayThemeCounts={isEssayView ? essayThemeCounts : null}
        />
      </div>

      {/* Table — fills remaining space, scroll inside only */}
      <div className="px-6 pb-4 flex-1 min-h-0">
        <DataTable
          data={filteredSentences}
          essayData={isEssayView ? filteredEssays : undefined}
          viewMode={viewMode}
          onEssaySelect={setDetailEssayId}
        />
      </div>

      {/* Detail panel — slide-out from right */}
      {detailEssayId && (
        <EssayDetailPanel
          essayId={detailEssayId}
          onClose={() => setDetailEssayId(null)}
        />
      )}
    </div>
  );
}
