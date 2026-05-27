# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from dotenv import load_dotenv

from research_application.host.factory import create_research_backend_app

load_dotenv()

app = create_research_backend_app()


def run() -> None:
    import uvicorn

    host = os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RESEARCH_BACKEND_PORT", "8010"))
    uvicorn.run(
        "research_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("RESEARCH_BACKEND_RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
