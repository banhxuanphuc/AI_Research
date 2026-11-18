# AI Research Scientist

Research workspace có bằng chứng: tìm paper, parse PDF, trích PaperCard có evidence span, hybrid RAG có citation, reviewer kiểm chứng. Kế hoạch đầy đủ: [`../docs/AI_Research_Scientist_Build_Plan.md`](../docs/AI_Research_Scientist_Build_Plan.md).

## Chạy nhanh (Docker)

```powershell
Copy-Item .env.example .env   # rồi điền GEMINI_API_KEY (hoặc ANTHROPIC_API_KEY) và đổi các mật khẩu
docker compose up -d --build
```

Chọn LLM bằng `LLM_DEFAULT_MODEL` trong `.env`. Provider được suy ra từ tên model: `gemini-*`/`gemma-*` → Google, `claude-*` → Anthropic, `ollama:*` → Ollama local. Hiện đang dùng `gemini-4-hydrogen` (chạy được với gói free của key hiện tại). Đổi model xong chạy `docker compose up -d --force-recreate api worker`.

| URL | Nội dung |
|---|---|
| http://localhost:3000 | Web UI |
| http://localhost:8000/docs | OpenAPI / Swagger của API |
| http://localhost:8000/health/ready | Trạng thái từng dependency |
| http://localhost:6333/dashboard | Qdrant dashboard |

Bật thêm dịch vụ AI (GROBID + embedding bge-m3, cần thêm ~4–6 GB RAM):

```powershell
docker compose --profile ai up -d
```

Các profile khác: `rerank` (bge-reranker), `graph` (Neo4j), `ollama` (provider LLM thứ 2).

## Lưu ý môi trường (máy hiện tại)

- **Antivirus giải mã HTTPS (Norton Web/Mail Shield):** container không tin CA của Norton. Đặt root CA dạng PEM vào `infra/docker/certs/` (xem README trong thư mục đó). Không có file này thì `uv sync`, `npm ci` và các lời gọi API ra ngoài sẽ lỗi chứng chỉ.
- **MinIO:** MinIO đã ngừng phát hành image community, nên dự án dùng `bitnamilegacy/minio` (không còn cập nhật, không có web console). Có thể thay bằng bất kỳ server S3-compatible nào.
- **Qdrant:** giữ server cùng minor với `qdrant-client` trong `uv.lock`. Storage chỉ nâng được từng minor một.
- **RAM:** Docker VM đang được cấp ~3.7 GB. Profile `ai` cần thêm ~4–6 GB, nên phải tăng bộ nhớ cho WSL (`%UserProfile%\.wslconfig`, mục `memory=`) trước khi bật.
- **Semantic Scholar:** không có API key thì hay bị 429. Response của `/discovery/search` có trường `source_errors` để biết nguồn nào lỗi.

## Test

Máy này có Application Control chặn `python.exe` ngoài hệ thống, nên test chạy trong container:

```powershell
docker compose run --rm --no-deps api pytest -q
```

## Migration

```powershell
docker compose run --rm api alembic revision --autogenerate -m "mo ta thay doi"
docker compose run --rm api alembic upgrade head
```

## Cấu trúc

```
src/research_scientist/
  api/          FastAPI app + routers (projects, papers, discovery, runs, health)
  worker/       arq worker (ingest PDF -> parse -> chunk -> embed)
  core/         config, db, logging
  domain/       SQLAlchemy models (papers, chunks, claims, runs, llm_calls...)
  schemas/      Pydantic: PaperCard, ResearchBrief, PaperCandidate
  providers/    LLM gateway (routing, budget, refusal fallback), Claude, Ollama, TEI
  ingestion/    Semantic Scholar / arXiv / Crossref, dedup DOI, GROBID, PyMuPDF, chunking, MinIO
  retrieval/    Qdrant, Postgres full-text, RRF fusion
  agents/       Brief Planner, Paper Analyst, evidence verifier
  workflows/    LangGraph graph có human approval (interrupt)
  security/     untrusted-content wrapper, injection heuristics, URL allowlist
  evals/        Recall@K, MRR, nDCG, citation precision
  prompts/      prompt có version (*.v1.md)
apps/web/       Next.js 16 + Tailwind + TanStack Query
datasets/eval/  golden sets (tự gán nhãn ở Sprint 0)
migrations/     Alembic
compose.yaml    Docker Compose có profiles
```

## Luồng đã chạy được trong skeleton

1. `POST /projects`: tạo project.
2. `POST /discovery/search`: tìm paper từ Semantic Scholar + arXiv, rồi dedup.
3. `POST /projects/{id}/runs`: workflow LangGraph sinh Research Brief bằng Claude, rồi **dừng chờ duyệt**.
4. `POST /runs/{run_id}/resume` với `{"approved": true}`: Discovery chạy, rồi **dừng chờ chọn paper**.
5. `POST /runs/{run_id}/resume` với `{"selected_indices": [0, 2]}`: kết thúc.
6. `POST /projects/{id}/papers/upload`: PDF được lưu vào MinIO, worker parse (GROBID hoặc PyMuPDF), chunk theo section, embed và đưa vào Qdrant.
7. `GET /papers/{id}/chunks`: xem chunk đã parse.

## Việc tiếp theo theo lộ trình

- Tuần 1–2: chạy `extract_paper_card` trên paper thật, tạo bộ test 30–50 case, benchmark Opus 5.5 / Sonnet 5.5 / Haiku 5.5 / Ollama.
- Tuần 3–4: sparse vector + hybrid search + rerank + endpoint `/ask` có Citations.
- Tuần 5–6: Discovery agent có tool calling, approval UI trên web.
- Tuần 7–8: MCP server, Langfuse tracing, Ragas/DeepEval, test prompt injection.
