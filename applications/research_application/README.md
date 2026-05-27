# Research Application (prototype)

Thin execution environment for the research → summarize multi-agent pipeline.

```bash
uv run uvicorn research_application.host.main:app --host 0.0.0.0 --port 8010
```

POST `/v1/research/run` with JSON body `{ "message": "your research question" }`.
