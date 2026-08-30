import { PageContainer } from '../components/layout/PageContainer';
import { FilterBar } from '../components/data-explorer/FilterBar';
import { ThemeDistributionChart } from '../components/exploration/ThemeDistributionChart';
import { CoOccurrenceHeatmap } from '../components/exploration/CoOccurrenceHeatmap';
import { CardinalityChart } from '../components/exploration/CardinalityChart';
import { TemporalChart } from '../components/exploration/TemporalChart';
import { CourseChart } from '../components/exploration/CourseChart';
import { LengthChart } from '../components/exploration/LengthChart';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { useDataLoader } from '../hooks/useDataLoader';
import { useFilteredSentences } from '../hooks/useFilteredData';
import { exportSentencesCSV } from '../lib/csv';
import { useCallback } from 'react';

export function ExplorationPage() {
  const { isLoading } = useDataLoader();
  const { filtered, total } = useFilteredSentences();

  const handleExport = useCallback(() => {
    exportSentencesCSV(filtered, 'alma_exploration_subset.csv');
  }, [filtered]);

  if (isLoading) {
    return (
      <PageContainer>
        <LoadingSpinner message="Loading data for exploration..." />
      </PageContainer>
    );
  }

  const annotatedCount = filtered.filter((s) => s.tags.annotated).length;

  return (
    <PageContainer>
      <div className="mb-4">
        <h1 className="text-lg font-semibold text-gray-900 m-0">Research & Exploration</h1>
        <p className="text-[13px] text-gray-500 mt-0.5">
          Build data subsets with filters, then visualize distributions, co-occurrence, and trends
        </p>
      </div>

      <div className="space-y-4">
        <FilterBar
          filteredCount={filtered.length}
          totalCount={total}
          onExport={handleExport}
        />

        {/* Summary cards */}
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Sentences', value: filtered.length },
            { label: 'Annotated', value: annotatedCount },
            { label: 'Unannotated', value: filtered.length - annotatedCount },
            { label: 'Essays', value: new Set(filtered.map((s) => s.essay_id)).size },
          ].map((card) => (
            <div key={card.label} className="border border-gray-200 rounded-lg p-3 bg-white">
              <p className="text-[11px] text-gray-500 uppercase tracking-wider m-0">{card.label}</p>
              <p className="text-xl font-semibold font-mono text-gray-900 mt-1 m-0">
                {card.value.toLocaleString()}
              </p>
            </div>
          ))}
        </div>

        {/* Charts grid */}
        <div className="grid grid-cols-2 gap-4">
          <ThemeDistributionChart data={filtered} />
          <CoOccurrenceHeatmap data={filtered} />
          <CardinalityChart data={filtered} />
          <LengthChart data={filtered} />
        </div>

        <div className="grid grid-cols-1 gap-4">
          <TemporalChart data={filtered} groupBy="year" />
          <CourseChart data={filtered} />
        </div>
      </div>
    </PageContainer>
  );
}
