// Client gọi FastAPI. Tuần sau: sinh type tự động từ OpenAPI (openapi-typescript).
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Readiness = {
  status: "ready" | "degraded";
  llm_default_model: string;
  anthropic_key_configured: boolean;
  gemini_key_configured: boolean;
  checks: Record<string, boolean>;
};

export type Project = {
  id: string;
  title: string;
  topic: string;
  brief: Record<string, unknown> | null;
  created_at: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  readiness: () => request<Readiness>("/health/ready"),
  listProjects: () => request<Project[]>("/projects"),
  createProject: (body: { title: string; topic: string }) =>
    request<Project>("/projects", { method: "POST", body: JSON.stringify(body) }),
};
