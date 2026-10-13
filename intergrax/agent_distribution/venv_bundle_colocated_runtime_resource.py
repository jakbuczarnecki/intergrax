# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Physical in-process runtime resource for co-located VENV_BUNDLE units (ADR-01 A1)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

_QUALIFICATION_WARMUP_MODULE = "bundle_marker"


@dataclass(slots=True)
class ColocatedVenvRuntimeResource:
    """Initialized import surface for one immutable bundle site-packages tree."""

    site_packages: Path
    artifact_digest: str
    warmup_module: str = _QUALIFICATION_WARMUP_MODULE
    _warmup_evidence: str | None = field(default=None, repr=False)
    _loaded_scoped_name: str | None = field(default=None, repr=False)

    @property
    def is_live(self) -> bool:
        return self._warmup_evidence is not None

    @property
    def warmup_evidence(self) -> str | None:
        return self._warmup_evidence

    def initialize(self) -> str:
        """Load and execute the qualification warmup module from site-packages."""
        if self._warmup_evidence is not None:
            return self._warmup_evidence

        resolved_site = self.site_packages.resolve()
        marker_path = resolved_site / f"{self.warmup_module}.py"
        if not marker_path.is_file():
            raise RuntimeError(
                f"venv bundle runtime warmup module missing: {marker_path}"
            )

        scoped_name = (
            f"_intergrax_colocated_unit_{self.artifact_digest.removeprefix('sha256:')[:16]}"
            f".{self.warmup_module}"
        )
        spec = importlib.util.spec_from_file_location(scoped_name, marker_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot create import spec for warmup module {marker_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[scoped_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(scoped_name, None)
            raise
        if not isinstance(module, ModuleType):
            raise RuntimeError("warmup module did not materialize as a module")
        marker_value = getattr(module, "MARKER", None)
        if marker_value is None:
            self._teardown_scoped_module(scoped_name)
            raise RuntimeError("warmup module lacks MARKER export")

        evidence = (
            f"venv-colocated-runtime:site={resolved_site.name}:"
            f"marker={marker_value!s}:digest={self.artifact_digest}"
        )
        self._warmup_evidence = evidence
        self._loaded_scoped_name = scoped_name
        return evidence

    def invalidate(self) -> None:
        if self._loaded_scoped_name is not None:
            self._teardown_scoped_module(self._loaded_scoped_name)
        self._loaded_scoped_name = None
        self._warmup_evidence = None

    def _teardown_scoped_module(self, scoped_name: str) -> None:
        sys.modules.pop(scoped_name, None)


__all__ = ["ColocatedVenvRuntimeResource"]
