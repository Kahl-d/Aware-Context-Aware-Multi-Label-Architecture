import { THEME_COLORS, THEME_ABBREVIATIONS, type ThemeName } from '../../constants/themes';

interface ThemeBadgeProps {
  theme: ThemeName | 'Class_0';
  size?: 'sm' | 'md';
  showFull?: boolean;
}

export function ThemeBadge({ theme, size = 'sm', showFull = false }: ThemeBadgeProps) {
  const color = THEME_COLORS[theme];
  const label = showFull ? theme.replace('_', ' ') : THEME_ABBREVIATIONS[theme];

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full whitespace-nowrap ${
        size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs'
      }`}
      style={{
        backgroundColor: `${color}14`,
        color: color,
        border: `1px solid ${color}30`,
      }}
    >
      {label}
    </span>
  );
}
