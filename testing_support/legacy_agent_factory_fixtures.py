# © Artur Czarnecki. All rights reserved.

"""Legacy-path factory fixtures for architecture gates (zero-arg compatibility)."""

from __future__ import annotations

from echo.echo_agent import EchoAgent


def build_ac5_legacy_zero_arg_factory() -> EchoAgent:
    return EchoAgent()
