# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from attestation_demo.host.factory import create_attestation_demo_application

load_dotenv()

app = create_attestation_demo_application()


def run() -> None:
    import uvicorn

    host = os.environ.get("ATTESTATION_DEMO_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("ATTESTATION_DEMO_BACKEND_PORT", "8097"))
    uvicorn.run(
        "attestation_demo.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("ATTESTATION_DEMO_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
