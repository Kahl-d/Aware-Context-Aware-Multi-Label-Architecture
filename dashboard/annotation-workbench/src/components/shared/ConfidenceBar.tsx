import { THEME_COLORS, THEME_ABBREVIATIONS, type ThemeName } from '../../constants/themes';

interface ConfidenceBarProps {
  theme: ThemeName;
  probability: number;
  threshold: number;
  predicted: boolean;
}

export function ConfidenceBar({ theme, probability, threshold, predicted }: ConfidenceBarProps) {
  const color = THEME_COLORS[theme];
  const abbrev = THEME_ABBREVIATIONS[theme];
  const pct = Math.round(probability * 100);

  return (
    <div className="flex items-center gap-2 text-[12px]">
      <span className="w-7 text-right font-mono text-gray-500">{abbrev}</span>
      <div className="flex-1 h-5 bg-gray-100 rounded-sm relative overflow-hidden">
        <div
          className="h-full rounded-sm transition-all duration-300"
          style={{
            width: `${pct}%`,
            backgroundColor: predicted ? color : `${color}40`,
          }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-gray-400"
          style={{ left: `${threshold * 100}%` }}
        />
      </div>
      <span className="w-10 text-right font-mono text-gray-600">{probability.toFixed(2)}</span>
      {predicted && <span className="text-green-600 text-[11px]">&#10003;</span>}
      {!predicted && <span className="w-3" />}
    </div>
  );
}
