# © Artur Czarnecki. All rights reserved.

"""Injectable ports for mandatory vs best-effort shutdown phases (certification only)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecordingMandatoryEvidenceFlush:
    """Mandatory evidence flush — failure fails shutdown closed."""

    fail_on_call: int | None = None
    calls: int = 0
    flushed: bool = False

    async def flush_required_evidence(self) -> None:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("mandatory_evidence_flush_failed")
        self.flushed = True


@dataclass
class InMemoryFinalStateStore:
    """Final runtime state persistence before worker termination."""

    fail_on_call: int | None = None
    calls: int = 0
    persisted: bool = False
    payload: dict[str, str] = field(default_factory=dict)

    async def persist_final_state(self, payload: dict[str, str]) -> None:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("final_state_persistence_failed")
        self.payload = dict(payload)
        self.persisted = True


@dataclass
class RecordingObservabilityExporter:
    """Best-effort OTLP-style export — must not block mandatory persistence."""

    fail_on_call: int | None = None
    calls: int = 0
    closed: bool = False

    async def close_export(self) -> None:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("observability_export_close_failed")
        self.closed = True
