# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Backward-compatibility shim for ``import legal_agent``.

Prefer:
- ``legal`` — reusable agent capability module (Layer 3)
- ``legal_application`` — execution environment / host (applications)
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import List

_AGENT_SUBMODULES: List[str] = [
    "config",
    "domain",
    "governance",
    "memory",
    "pipeline",
    "prompts",
    "runtime",
    "steps",
    "tracing",
    "tests",
    "use_cases",
]

_APPLICATION_SUBMODULES: List[str] = [
    "host",
    "serving",
]

_INSTALLED = False


def _install_shim() -> ModuleType:
    global _INSTALLED
    existing = sys.modules.get("legal_agent")
    if _INSTALLED and existing is not None:
        return existing  # type: ignore[return-value]

    import legal  # noqa: WPS433

    pkg = ModuleType("legal_agent")
    pkg.__doc__ = __doc__
    pkg._intergrax_shim = True  # type: ignore[attr-defined]
    sys.modules["legal_agent"] = pkg

    sys.modules["legal_agent.legal_agent"] = legal.legal_agent
    pkg.legal_agent = legal.legal_agent  # type: ignore[attr-defined]

    for name in _AGENT_SUBMODULES:
        full = f"legal.{name}"
        mod = importlib.import_module(full)
        sys.modules[f"legal_agent.{name}"] = mod
        setattr(pkg, name, mod)

    for name in _APPLICATION_SUBMODULES:
        full = f"legal_application.{name}"
        mod = importlib.import_module(full)
        sys.modules[f"legal_agent.{name}"] = mod
        setattr(pkg, name, mod)

    _INSTALLED = True
    return pkg


def __getattr__(name: str):
    pkg = _install_shim()
    if name == "LegalAgent":
        return pkg.legal_agent.LegalAgent
    return getattr(pkg, name)


def __dir__() -> List[str]:
    pkg = _install_shim()
    return sorted(set(list(globals().keys()) + dir(pkg) + ["LegalAgent"]))


__all__ = ["LegalAgent"]
