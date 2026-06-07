# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from dispute_sim_application.host.factory import create_dispute_sim_backend_app

load_dotenv()

app = create_dispute_sim_backend_app()


def run() -> None:
    import uvicorn

    host = os.environ.get("DISPUTE_SIM_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("DISPUTE_SIM_BACKEND_PORT", "8020"))
    uvicorn.run(
        "dispute_sim_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("DISPUTE_SIM_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
