# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from dotenv import load_dotenv

from lab_application.host.factory import create_lab_application

load_dotenv()

app = create_lab_application()


def run() -> None:
    import uvicorn

    host = os.environ.get("LAB_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("LAB_BACKEND_PORT", "8090"))
    reload = os.environ.get("LAB_BACKEND_RELOAD", "").strip().lower() in {"1", "true", "yes"}
    uvicorn.run(
        "lab_application.host.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run()
