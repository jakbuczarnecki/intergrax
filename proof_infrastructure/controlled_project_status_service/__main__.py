# © Artur Czarnecki. All rights reserved.

"""Run the controlled Project Status proof service on loopback HTTP."""

from __future__ import annotations

import argparse

import uvicorn

from proof_infrastructure.controlled_project_status_service.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Project Status proof service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
