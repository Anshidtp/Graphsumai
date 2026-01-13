const API_BASE = "http://localhost:8000/api/v1";

export const fetchStats = async () => {
  const res = await fetch(`${API_BASE}/stats`);
  return res.json();
};

export const queryGraph = async (query, top_k = 15) => {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
  return res.json();
};
