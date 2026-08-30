import { THEMES, THEME_COLORS, type ThemeName } from '../../constants/themes';
import { useDataStore } from '../../stores/dataStore';
import type { SentenceRecord } from '../../types/data';

interface LabelEditorProps {
  sentence: SentenceRecord;
  onDone: () => void;
}

/**
 * Inline label editor for a single sentence.
 * Shows toggleable theme chips — click to add/remove a theme.
 * Toggling any theme ON clears Class_0; setting all OFF sets Class_0.
 */
export function LabelEditor({ sentence, onDone }: LabelEditorProps) {
  const annotatorName = useDataStore((s) => s.annotatorName);
  const updateLabel = useDataStore((s) => s.updateLabel);

  const activeThemes = THEMES.filter((t) => sentence.labels[t] === 1);
  const isClass0 = activeThemes.length === 0;

  function toggleTheme(theme: ThemeName) {
    if (!annotatorName) return;

    const currentVal = sentence.labels[theme];
    const newVal = currentVal === 1 ? 0 : 1;

    // Update the theme
    updateLabel(sentence.essay_id, sentence.sentence_id, theme, newVal, annotatorName);

    // If we just turned ON a theme, make sure Class_0 is 0
    if (newVal === 1) {
      updateLabel(sentence.essay_id, sentence.sentence_id, 'Class_0', 0, annotatorName);
    } else {
      // If we just turned OFF a theme, check if all themes are now off → set Class_0
      const remainingActive = THEMES.filter(
        (t) => t === theme ? false : sentence.labels[t] === 1
      );
      if (remainingActive.length === 0) {
        updateLabel(sentence.essay_id, sentence.sentence_id, 'Class_0', 1, annotatorName);
      }
    }
  }

  return (
    <div className="mt-2 p-2.5 bg-gray-50 rounded-md border border-gray-200 space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {THEMES.map((theme) => {
          const isActive = sentence.labels[theme] === 1;
          const color = THEME_COLORS[theme];
          return (
            <button
              key={theme}
              onClick={() => toggleTheme(theme)}
              className={`px-2 py-0.5 rounded-full text-[11px] font-medium border transition-all ${
                isActive
                  ? 'text-white border-transparent'
                  : 'text-gray-500 border-gray-200 bg-white hover:border-gray-300'
              }`}
              style={isActive ? { backgroundColor: color, borderColor: color } : {}}
              title={isActive ? `Remove ${theme}` : `Add ${theme}`}
            >
              {theme.replace('_', ' ')}
            </button>
          );
        })}
        {isClass0 && (
          <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-200 text-gray-600">
            Class_0
          </span>
        )}
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-gray-400">
          Editing as <span className="font-medium text-gray-600">{annotatorName}</span>
        </span>
        <button
          onClick={onDone}
          className="px-2.5 py-0.5 text-[11px] font-medium text-gray-600 bg-white border border-gray-200 rounded-md hover:bg-gray-50 transition-colors"
        >
          Done
        </button>
      </div>
    </div>
  );
}
