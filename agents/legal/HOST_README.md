# Legal backend host (moved)

The **Tier-3 FastAPI host** (product shell) now lives under:

`applications/legal_application/host/`

See [`applications/legal_application/host/README.md`](../../applications/legal_application/host/README.md).

This directory (`agents/legal/`) contains only the **Legal Agent capability module** (pipeline, steps, domain logic).

Backward-compatible imports still work via the `legal_agent` shim:

```bash
uv run uvicorn legal_agent.host.main:app --host 0.0.0.0 --port 8000
```

Preferred path:

```bash
uv run uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
```
