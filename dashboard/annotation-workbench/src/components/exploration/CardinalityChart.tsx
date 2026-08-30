import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { cardinalityDistribution } from '../../lib/stats';
import type { SentenceRecord } from '../../types/data';

interface Props {
  data: SentenceRecord[];
}

const COLORS = ['#6b7280', '#2563eb', '#0891b2', '#16a34a', '#ea580c'];

export function CardinalityChart({ data }: Props) {
  const chartData = useMemo(() => {
    const dist = cardinalityDistribution(data);
    return Object.entries(dist).map(([labels, count]) => ({
      labels: labels === '0' ? 'Class_0' : `${labels} theme${labels === '1' ? '' : 's'}`,
      count,
    }));
  }, [data]);

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-[13px] font-semibold text-gray-700 mb-3 m-0">Label Cardinality</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="labels" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
