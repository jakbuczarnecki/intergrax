# Legal Agent — backward-compatibility shim

This directory exists **only** for legacy import paths (`legal_agent.*`).

## Use instead

| Legacy | Canonical |
|--------|-----------|
| `legal_agent.legal_agent` | `legal.legal_agent` |
| `legal_agent.host` | `legal_application.host` |
| `legal_agent.serving` | `legal_application.serving` |

Agent capability code: `agents/legal/`  
Application host/serving: `applications/legal_application/`

The package `legal_agent` re-exports modules at import time via PEP 562 (`__init__.py`).
