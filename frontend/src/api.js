// Change this if your backend runs somewhere else:
export const API_BASE =
  import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export async function postCSV(endpoint, file, outAsBlob = false) {
  const fd = new FormData();
  fd.append('file', file);
  const resp = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    body: fd,
  });
  if (!resp.ok) {
    let detail = '';
    try { detail = (await resp.json()).detail || ''; } catch {}
    throw new Error(detail || `HTTP ${resp.status}`);
  }
  return outAsBlob ? resp.blob() : resp.json();
}

export async function getJSON(endpoint) {
  const resp = await fetch(`${API_BASE}${endpoint}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}
