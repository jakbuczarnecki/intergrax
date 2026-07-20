# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from governed_contractor_application.host.factory import create_governed_contractor_backend_app

load_dotenv()

app = create_governed_contractor_backend_app()


def run() -> None:
    import uvicorn

    host = os.environ.get("GOVERNED_CONTRACTOR_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("GOVERNED_CONTRACTOR_BACKEND_PORT", "8000"))
    uvicorn.run(
        "governed_contractor_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("GOVERNED_CONTRACTOR_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
