export function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-3">
      <div className="w-6 h-6 border-2 border-gray-200 border-t-gray-600 rounded-full animate-spin" />
      <p className="text-sm text-gray-500">{message}</p>
    </div>
  );
}
