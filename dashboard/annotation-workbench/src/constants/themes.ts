export const THEMES = [
  'Attainment',
  'Aspirational',
  'Navigational',
  'Resistance',
  'Perseverance',
  'Social',
  'Spiritual',
  'Familial_Capital',
] as const;

export type ThemeName = (typeof THEMES)[number];

export const THEME_COLORS: Record<ThemeName | 'Class_0', string> = {
  Attainment: '#7c3aed',
  Aspirational: '#2563eb',
  Navigational: '#0891b2',
  Resistance: '#dc2626',
  Perseverance: '#ea580c',
  Social: '#16a34a',
  Spiritual: '#a855f7',
  Familial_Capital: '#ca8a04',
  Class_0: '#6b7280',
};

export const THEME_ABBREVIATIONS: Record<ThemeName | 'Class_0', string> = {
  Attainment: 'ATT',
  Aspirational: 'ASP',
  Navigational: 'NAV',
  Resistance: 'RES',
  Perseverance: 'PER',
  Social: 'SOC',
  Spiritual: 'SPI',
  Familial_Capital: 'FAM',
  Class_0: 'C0',
};

export const THEME_BG_COLORS: Record<ThemeName | 'Class_0', string> = {
  Attainment: '#f5f3ff',
  Aspirational: '#eff6ff',
  Navigational: '#ecfeff',
  Resistance: '#fef2f2',
  Perseverance: '#fff7ed',
  Social: '#f0fdf4',
  Spiritual: '#faf5ff',
  Familial_Capital: '#fefce8',
  Class_0: '#f9fafb',
};
