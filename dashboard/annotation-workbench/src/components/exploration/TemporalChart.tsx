import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { THEME_COLORS, THEMES } from '../../constants/themes';
import { themesByGroup } from '../../lib/stats';
import type { SentenceRecord } from '../../types/data';

interface Props {
  data: SentenceRecord[];
  groupBy?: 'year' | 'semester' | 'course';
}

export function TemporalChart({ data, groupBy = 'year' }: Props) {
  const chartData = useMemo(() => themesByGroup(data, groupBy), [data, groupBy]);

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[13px] font-semibold text-gray-700 mb-3 m-0">
        Theme Trends by {groupBy.charAt(0).toUpperCase() + groupBy.slice(1)}
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="group" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {THEMES.map((theme) => (
            <Line
              key={theme}
              type="monotone"
              dataKey={theme}
              stroke={THEME_COLORS[theme]}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
