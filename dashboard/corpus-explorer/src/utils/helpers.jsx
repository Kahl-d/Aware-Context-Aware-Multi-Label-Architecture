import { THEME_COLORS } from '../constants/themes'

export function highlightText(text, searchTerm) {
  if (!searchTerm || !text) return text
  const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  const parts = text.split(regex)
  return parts.map((part, i) => regex.test(part) ? <mark key={i}>{part}</mark> : part)
}

export function fmt(n) {
  if (n == null) return '--'
  return typeof n === 'number' ? n.toLocaleString() : n
}

export function pct(n, total) {
  if (!total) return '0%'
  return `${(n / total * 100).toFixed(1)}%`
}

export function themeColor(theme) {
  return THEME_COLORS[theme] || '#94a3b8'
}

// Normalize essay data fields (handle inconsistent top-level vs metadata)
export function normalizeEssay(e) {
  return {
    ...e,
    year: e.year || e.metadata?.year || 'Unknown',
    semester: e.semester || e.metadata?.semester || 'Unknown',
    class_name: e.class_name || e.metadata?.class_name || 'Unknown',
    data_source: e.data_source || e.metadata?.data_source || 'Unknown',
  }
}
