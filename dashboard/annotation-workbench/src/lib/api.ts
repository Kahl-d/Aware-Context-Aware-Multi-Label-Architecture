const API_BASE = '/api';

export async function inferSingle(text: string, modelId: string) {
  const res = await fetch(`${API_BASE}/infer/single`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function inferBatch(file: File, modelId: string) {
  const form = new FormData();
  form.append('file', file);
  form.append('model_id', modelId);

  const res = await fetch(`${API_BASE}/infer/batch`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res;
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Could not reach inference server');
  return res.json();
}
