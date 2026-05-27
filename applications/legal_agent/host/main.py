# © Artur Czarnecki. All rights reserved.
# Shim — use ``legal_application.host.main`` in new code.

from legal_application.host.main import app, run

__all__ = ["app", "run"]
