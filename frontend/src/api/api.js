// src/api/api.js
import api from '@/services/axiosInstance';

// Generic helpers
export const getJSON = (path) => api.get(path).then(r => r.data);
export const postJSON = (path, data) => api.post(path, data).then(r => r.data);

// Single CSV upload (returns JSON or Blob)
export async function postCSV(path, file, { fieldName = 'file', expectBlob = false } = {}) {
  const fd = new FormData();
  fd.append(fieldName, file);
  const res = await api.post(path, fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: expectBlob ? 'blob' : 'json',
  });
  return res.data;
}

// Two CSVs (evaluate)
export async function postTwoCSVs(path, files) {
  const fd = new FormData();
  if (files.actual) fd.append('actual', files.actual);
  if (files.pred) fd.append('pred', files.pred);
  const res = await api.post(path, fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

// Download helper
export function downloadBlob(blob, filename = 'download.csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(a.href);
  a.remove();
}

// ---- Your backend API ----
export const health = () => getJSON('/api/health');
export const featureImportances = () => getJSON('/api/feature_importances');

export const trainCSV = (file) => postCSV('/api/train_csv', file); // JSON metrics
export const predictCSV = (file) => postCSV('/api/predict_csv', file, { expectBlob: true });
export const checkExoCSV = (file) => postCSV('/api/check_exo_csv', file, { expectBlob: true });
export const checkExoStatusCSV = (file) => postCSV('/api/check_exo_status_csv', file, { expectBlob: true });
export const evaluatePairCSV = (actualFile, predFile) => postTwoCSVs('/api/evaluate_pair_csv', { actual: actualFile, pred: predFile });

export const predictRows = (rows) => postJSON('/api/predict', { rows });
export const koiStatus = (payload) => postJSON('/api/koi_status', payload);
