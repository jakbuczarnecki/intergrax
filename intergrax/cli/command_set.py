# © Artur Czarnecki. All rights reserved.

"""Tier-0 seam for optional injected CLI command sets (no Tier-2/3 names)."""

from __future__ import annotations

import argparse
from typing import Protocol


class CliCommandSet(Protocol):
    def register_parsers(self, sub: argparse._SubParsersAction) -> None:
        """Register command parsers; each leaf must set ``cli_run`` on defaults."""


__all__ = ["CliCommandSet"]
