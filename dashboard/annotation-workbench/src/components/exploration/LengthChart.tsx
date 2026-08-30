import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { lengthDistribution } from '../../lib/stats';
import type { SentenceRecord } from '../../types/data';

interface Props {
  data: SentenceRecord[];
}

export function LengthChart({ data }: Props) {
  const chartData = useMemo(() => lengthDistribution(data), [data]);

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[13px] font-semibold text-gray-700 mb-3 m-0">Sentence Length Distribution</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="bucket" tick={{ fontSize: 11 }} label={{ value: 'words', position: 'insideBottom', offset: -2, fontSize: 10 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
          <Bar dataKey="count" fill="#6b7280" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
