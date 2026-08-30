import { useMemo } from 'react';
import { THEME_ABBREVIATIONS, THEMES } from '../../constants/themes';
import { coOccurrenceMatrix } from '../../lib/stats';
import type { SentenceRecord } from '../../types/data';

interface Props {
  data: SentenceRecord[];
}

export function CoOccurrenceHeatmap({ data }: Props) {
  const { matrix } = useMemo(() => coOccurrenceMatrix(data), [data]);

  const maxVal = useMemo(() => {
    let max = 0;
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        if (i !== j && matrix[i][j] > max) max = matrix[i][j];
      }
    }
    return max || 1;
  }, [matrix]);

  const cellSize = 38;
  const labelWidth = 36;
  const size = THEMES.length * cellSize + labelWidth;

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[13px] font-semibold text-gray-700 mb-3 m-0">Theme Co-occurrence</h3>
      <div className="overflow-auto">
        <svg width={size} height={size} className="mx-auto">
          {/* Column labels */}
          {THEMES.map((t, j) => (
            <text
              key={`col-${j}`}
              x={labelWidth + j * cellSize + cellSize / 2}
              y={labelWidth - 6}
              textAnchor="middle"
              fontSize={10}
              fill="#6b7280"
            >
              {THEME_ABBREVIATIONS[t]}
            </text>
          ))}
          {/* Row labels + cells */}
          {THEMES.map((t, i) => (
            <g key={`row-${i}`}>
              <text
                x={labelWidth - 4}
                y={labelWidth + i * cellSize + cellSize / 2 + 4}
                textAnchor="end"
                fontSize={10}
                fill="#6b7280"
              >
                {THEME_ABBREVIATIONS[t]}
              </text>
              {THEMES.map((_t2, j) => {
                const val = matrix[i][j];
                const intensity = i === j ? 0.15 : Math.min(val / maxVal, 1);
                return (
                  <g key={`cell-${i}-${j}`}>
                    <rect
                      x={labelWidth + j * cellSize + 1}
                      y={labelWidth + i * cellSize + 1}
                      width={cellSize - 2}
                      height={cellSize - 2}
                      rx={3}
                      fill={
                        i === j
                          ? `rgba(107, 114, 128, ${intensity})`
                          : `rgba(37, 99, 235, ${intensity * 0.8})`
                      }
                    />
                    {val > 0 && (
                      <text
                        x={labelWidth + j * cellSize + cellSize / 2}
                        y={labelWidth + i * cellSize + cellSize / 2 + 4}
                        textAnchor="middle"
                        fontSize={9}
                        fill={intensity > 0.5 ? '#fff' : '#374151'}
                        fontFamily="var(--font-mono)"
                      >
                        {val}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
