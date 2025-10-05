// src/services/axiosInstance.js
import axios from 'axios';

const base =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE) ||
  'http://192.168.3.41:8000';

const api = axios.create({
  baseURL: base,
  timeout: 120000,
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err?.response?.data?.detail ||
      err?.response?.data?.error ||
      err?.message ||
      'Request failed';
    return Promise.reject(new Error(msg));
  }
);

export default api;
