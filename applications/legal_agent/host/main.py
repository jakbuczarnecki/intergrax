# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ASGI entrypoint for the Legal backend host."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from legal_agent.host.factory import create_legal_backend_app

# Load `.env` when present (does not override existing process env).
load_dotenv()

app = create_legal_backend_app()


def run() -> None:
    """CLI entry (`python -m legal_agent.host.main`) using uvicorn when installed."""
    import uvicorn

    host = os.environ.get("LEGAL_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("LEGAL_BACKEND_PORT", "8000"))
    reload = os.environ.get("LEGAL_BACKEND_RELOAD", "").strip().lower() in {"1", "true", "yes"}
    uvicorn.run(
        "legal_agent.host.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run()
