import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { THEME_COLORS, THEME_ABBREVIATIONS, THEMES, type ThemeName } from '../../constants/themes';
import { themeDistribution } from '../../lib/stats';
import type { SentenceRecord } from '../../types/data';

interface Props {
  data: SentenceRecord[];
  title?: string;
}

export function ThemeDistributionChart({ data, title = 'Theme Distribution' }: Props) {
  const chartData = useMemo(() => {
    const dist = themeDistribution(data);
    return [...THEMES, 'Class_0' as const].map((t) => ({
      theme: THEME_ABBREVIATIONS[t as ThemeName | 'Class_0'],
      fullName: t.replace('_', ' '),
      count: dist[t] || 0,
      color: THEME_COLORS[t as ThemeName | 'Class_0'],
    }));
  }, [data]);

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[13px] font-semibold text-gray-700 mb-3 m-0">{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="theme" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(value: number) => [value.toLocaleString(), 'count']}
            contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
