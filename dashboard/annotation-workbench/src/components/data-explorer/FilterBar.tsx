import { useDataStore } from '../../stores/dataStore';
import { useFilterOptions } from '../../hooks/useFilteredData';
import { useDebounce } from '../../hooks/useDebounce';
import { THEMES } from '../../constants/themes';
import { useState, useEffect } from 'react';

interface FilterBarProps {
  filteredCount: number;
  totalCount: number;
  onExport?: () => void;
  /** Essay-level counts when in essay view mode */
  essayThemeCounts?: { themeCounts: Record<string, number>; class0Count: number; totalEssays: number } | null;
}

function Select({
  label,
  value,
  onChange,
  options,
  allLabel = 'All',
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  allLabel?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 px-2 text-[13px] border border-gray-200 rounded-md bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-gray-300 focus:border-gray-300"
      >
        <option value="">{allLabel}</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

export function FilterBar({ filteredCount, totalCount, onExport, essayThemeCounts }: FilterBarProps) {
  const filters = useDataStore((s) => s.filters);
  const setFilter = useDataStore((s) => s.setFilter);
  const resetFilters = useDataStore((s) => s.resetFilters);
  const { courses, semesters, years, prompts, coders } = useFilterOptions();

  const [searchInput, setSearchInput] = useState(filters.search);
  const debouncedSearch = useDebounce(searchInput, 300);

  useEffect(() => {
    setFilter('search', debouncedSearch);
  }, [debouncedSearch, setFilter]);

  const viewMode = filters.viewMode;
  const isEssayView = viewMode === 'essay';

  return (
    <div className="border border-gray-200 rounded-lg bg-gray-50/50 p-4 space-y-3">
      {/* Row 0: View mode toggle */}
      <div className="flex items-center gap-3">
        <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">View</span>
        <div className="flex rounded-md border border-gray-200 overflow-hidden">
          <button
            onClick={() => setFilter('viewMode', 'sentence')}
            className={`px-3 py-1 text-[12px] font-medium transition-colors ${
              !isEssayView
                ? 'bg-gray-800 text-white'
                : 'bg-white text-gray-500 hover:bg-gray-50'
            }`}
          >
            Sentences
          </button>
          <button
            onClick={() => setFilter('viewMode', 'essay')}
            className={`px-3 py-1 text-[12px] font-medium transition-colors border-l border-gray-200 ${
              isEssayView
                ? 'bg-gray-800 text-white'
                : 'bg-white text-gray-500 hover:bg-gray-50'
            }`}
          >
            Essays
          </button>
        </div>
        {isEssayView && (
          <span className="text-[11px] text-gray-400">
            Theme counts are per-essay (each theme counted once per essay)
          </span>
        )}
      </div>

      {/* Row 1: Primary filters */}
      <div className="flex flex-wrap gap-3">
        <Select
          label="Data"
          value={filters.annotatedStatus}
          onChange={(v) => setFilter('annotatedStatus', v as 'all' | 'annotated' | 'unannotated')}
          options={['annotated', 'unannotated']}
          allLabel="All data"
        />
        <Select
          label="Split"
          value={filters.split}
          onChange={(v) => setFilter('split', v as 'all' | 'train' | 'val' | 'test')}
          options={['train', 'val', 'test']}
          allLabel="All"
        />
        <Select
          label="Course"
          value={filters.course}
          onChange={(v) => setFilter('course', v)}
          options={courses}
        />
        <Select
          label="Semester"
          value={filters.semester}
          onChange={(v) => setFilter('semester', v)}
          options={semesters}
        />
        <Select
          label="Year"
          value={filters.year}
          onChange={(v) => setFilter('year', v)}
          options={years}
        />
        <Select
          label="Prompt"
          value={filters.prompt}
          onChange={(v) => setFilter('prompt', v)}
          options={prompts}
        />
        <Select
          label="Coder"
          value={filters.coder}
          onChange={(v) => setFilter('coder', v)}
          options={coders}
        />
      </div>

      {/* Row 2: Theme chips */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mr-1">
          Themes
        </span>
        {THEMES.map((theme) => {
          const active = filters.themes.includes(theme);
          return (
            <button
              key={theme}
              onClick={() => {
                const next = active
                  ? filters.themes.filter((t) => t !== theme)
                  : [...filters.themes, theme];
                setFilter('themes', next);
              }}
              className={`px-2 py-0.5 rounded-full text-[11px] font-medium border transition-colors ${
                active
                  ? 'bg-gray-800 text-white border-gray-800'
                  : 'bg-white text-gray-500 border-gray-200 hover:border-gray-300'
              }`}
            >
              {theme.replace('_', ' ')}
            </button>
          );
        })}
      </div>

      {/* Row 3: Search + stats + actions */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search essay ID, ALMA ID, or sentence text..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="w-full h-8 pl-8 pr-3 text-[13px] border border-gray-200 rounded-md bg-white text-gray-700 placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-300"
          />
          <svg
            className="absolute left-2.5 top-2 w-3.5 h-3.5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        <span className="text-[13px] text-gray-500">
          <span className="font-mono font-medium text-gray-700">
            {filteredCount.toLocaleString()}
          </span>{' '}
          of {totalCount.toLocaleString()} {isEssayView ? 'essays' : 'sentences'}
        </span>

        <div className="ml-auto flex items-center gap-2">
          {onExport && (
            <button
              onClick={onExport}
              className="px-3 h-8 text-[12px] font-medium text-gray-600 border border-gray-200 rounded-md bg-white hover:bg-gray-50 transition-colors"
            >
              Export CSV
            </button>
          )}
          <button
            onClick={resetFilters}
            className="px-3 h-8 text-[12px] font-medium text-gray-500 border border-gray-200 rounded-md bg-white hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Row 4: Essay-level theme frequency counts (essay view only) */}
      {isEssayView && essayThemeCounts && (
        <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-gray-200">
          <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mr-1">
            Essay Freq
          </span>
          {THEMES.map((theme) => {
            const count = essayThemeCounts.themeCounts[theme] || 0;
            if (count === 0) return null;
            return (
              <span
                key={theme}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-100 text-gray-600 border border-gray-200"
              >
                <span className="font-medium">{theme.replace('_', ' ')}</span>
                <span className="font-mono text-gray-800">{count}</span>
              </span>
            );
          })}
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-100 text-gray-500 border border-gray-200">
            Class_0 only
            <span className="font-mono text-gray-800">{essayThemeCounts.class0Count}</span>
          </span>
        </div>
      )}
    </div>
  );
}
