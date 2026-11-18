"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

import { api } from "@/lib/api";

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
        ok
          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
          : "bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
      }`}
    >
      <span className={`size-1.5 rounded-full ${ok ? "bg-emerald-500" : "bg-zinc-400"}`} />
      {label}
    </span>
  );
}

export default function Home() {
  const qc = useQueryClient();
  const ready = useQuery({ queryKey: ["ready"], queryFn: api.readiness, refetchInterval: 15000 });
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const [title, setTitle] = useState("");
  const [topic, setTopic] = useState("");

  const create = useMutation({
    mutationFn: api.createProject,
    onSuccess: () => {
      setTitle("");
      setTopic("");
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (title.trim() && topic.trim()) create.mutate({ title, topic });
  }

  return (
    <main className="mx-auto max-w-4xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">AI Research Scientist</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Research workspace có bằng chứng: mọi kết luận trỏ về paper, trang và đoạn cụ thể.
        </p>
      </header>

      <section className="mb-8 rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
        <h2 className="mb-3 text-sm font-medium text-zinc-500">Trạng thái hệ thống</h2>
        {ready.isError && <p className="text-sm text-red-600">Không kết nối được API.</p>}
        {ready.data && (
          <div className="flex flex-wrap gap-2">
            {Object.entries(ready.data.checks).map(([name, ok]) => (
              <StatusBadge key={name} ok={ok} label={name} />
            ))}
            <StatusBadge ok={ready.data.anthropic_key_configured} label="anthropic key" />
            <StatusBadge ok={ready.data.gemini_key_configured} label="gemini key" />
          </div>
        )}
        {ready.data && (
          <p className="mt-3 text-xs text-zinc-500">
            LLM mặc định: <code className="font-mono">{ready.data.llm_default_model}</code>
          </p>
        )}
      </section>

      <section className="mb-8 rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
        <h2 className="mb-3 text-sm font-medium text-zinc-500">Tạo research project</h2>
        <form onSubmit={onSubmit} className="grid gap-3">
          <input
            className="rounded-lg border border-zinc-300 bg-transparent px-3 py-2 text-sm dark:border-zinc-700"
            placeholder="Tên project, vd: Long-tailed image classification"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <textarea
            className="min-h-20 rounded-lg border border-zinc-300 bg-transparent px-3 py-2 text-sm dark:border-zinc-700"
            placeholder="Chủ đề, mục tiêu, phạm vi năm..."
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
          <button
            type="submit"
            disabled={create.isPending}
            className="justify-self-start rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
          >
            {create.isPending ? "Đang tạo..." : "Tạo project"}
          </button>
          {create.isError && <p className="text-sm text-red-600">{String(create.error)}</p>}
        </form>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium text-zinc-500">Projects</h2>
        {projects.data?.length === 0 && <p className="text-sm text-zinc-500">Chưa có project nào.</p>}
        <ul className="grid gap-2">
          {projects.data?.map((p) => (
            <li key={p.id} className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <p className="font-medium">{p.title}</p>
              <p className="mt-0.5 line-clamp-2 text-sm text-zinc-500">{p.topic}</p>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
