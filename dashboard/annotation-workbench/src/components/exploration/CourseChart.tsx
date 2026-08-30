import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { THEME_COLORS, THEMES } from '../../constants/themes';
import { themesByGroup } from '../../lib/stats';
import type { SentenceRecord } from '../../types/data';

interface Props {
  data: SentenceRecord[];
}

export function CourseChart({ data }: Props) {
  const chartData = useMemo(() => themesByGroup(data, 'course'), [data]);

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[13px] font-semibold text-gray-700 mb-3 m-0">Course Breakdown</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="group" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={60} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {THEMES.map((theme) => (
            <Bar key={theme} dataKey={theme} stackId="a" fill={THEME_COLORS[theme]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
