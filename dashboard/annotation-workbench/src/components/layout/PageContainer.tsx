import type { ReactNode } from 'react';

interface PageContainerProps {
  children: ReactNode;
  className?: string;
}

export function PageContainer({ children, className = '' }: PageContainerProps) {
  return (
    <main className={`max-w-[1400px] mx-auto px-6 py-6 w-full ${className}`}>
      {children}
    </main>
  );
}
