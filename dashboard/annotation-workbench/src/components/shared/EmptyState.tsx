interface EmptyStateProps {
  title: string;
  description?: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-2">
      <p className="text-sm font-medium text-gray-600">{title}</p>
      {description && <p className="text-xs text-gray-400">{description}</p>}
    </div>
  );
}
