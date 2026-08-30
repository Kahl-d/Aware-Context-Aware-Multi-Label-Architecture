import { useEffect } from 'react';
import { useDataStore } from '../stores/dataStore';
import type { SentenceRecord, EssayRecord } from '../types/data';

export function useDataLoader() {
  const { setSentences, setEssays, setLoading, setError, isLoading, sentences } = useDataStore();

  useEffect(() => {
    if (sentences.length > 0) return; // already loaded

    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const [sentRes, essayRes] = await Promise.all([
          fetch('/data/sentences.json'),
          fetch('/data/essays.json'),
        ]);

        if (!sentRes.ok || !essayRes.ok) {
          throw new Error('Failed to load data files');
        }

        const sentData: SentenceRecord[] = await sentRes.json();
        const essayData: EssayRecord[] = await essayRes.json();

        if (!cancelled) {
          setSentences(sentData);
          setEssays(essayData);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unknown error');
          setLoading(false);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [sentences.length, setSentences, setEssays, setLoading, setError]);

  return { isLoading, error: useDataStore((s) => s.error) };
}
