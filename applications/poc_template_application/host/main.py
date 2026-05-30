# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from poc_template_application.host.factory import create_poc_template_application

load_dotenv()

app = create_poc_template_application()


def run() -> None:
    import uvicorn

    host = os.environ.get("POC_TEMPLATE_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("POC_TEMPLATE_BACKEND_PORT", "8095"))
    uvicorn.run(
        "poc_template_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("POC_TEMPLATE_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
