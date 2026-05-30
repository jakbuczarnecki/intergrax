# Tier-3 applications

Configured execution environments that compose Nexus, agents, and integrations for HTTP/Docker deployment.

**Usage guide:** [`USAGE.md`](USAGE.md)  
**Composition engine:** [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)

| Application | Role |
|-------------|------|
| [`lab_application/`](lab_application/) | Universal lab + debug API |
| [`legal_application/`](legal_application/) | Legal product host |
| [`research_application/`](research_application/) | Research pipeline host |

Each application includes `manifest.py`, `host/`, `serving/`, `mcp/`, `.env.example`, `BUILD_AND_DEPLOY.md`, `docker/`, and `<app>_tests/` (or `legal_tests/` for legal). FastMCP is coupled to FastAPI via `intergrax.applications._shared.fastapi_mcp`.
