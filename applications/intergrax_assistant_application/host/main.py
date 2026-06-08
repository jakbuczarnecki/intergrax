# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from intergrax_assistant_application.host.factory import create_intergrax_assistant_application

load_dotenv()

app = create_intergrax_assistant_application()


def run() -> None:
    import uvicorn

    host = os.environ.get("INTERGRAX_ASSISTANT_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("INTERGRAX_ASSISTANT_BACKEND_PORT", "8096"))
    uvicorn.run(
        "intergrax_assistant_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("INTERGRAX_ASSISTANT_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
