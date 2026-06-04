# © Artur Czarnecki. All rights reserved.

"""Minimal external harness project entry (Phase DX-6.3)."""

from __future__ import annotations

from echo.echo_agent import EchoAgent
from intergrax.harness import AgentGraph, HarnessApplication
from intergrax.integrations.registry.profile import IntegrationProfile


def create_app():
    return (
        HarnessApplication("demo", route_prefix="/v1/demo")
        .agents(EchoAgent)
        .integrations(IntegrationProfile.lab_stack())
        .graph(AgentGraph().default(EchoAgent))
        .mode("balanced")
        .build_fastapi()
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8091, reload=True)
