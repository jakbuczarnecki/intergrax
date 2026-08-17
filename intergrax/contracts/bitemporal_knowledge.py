# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical bitemporal knowledge contracts (OBSERVABILITY §8, TRACE-BITEMP-1).

These types freeze valid-time / system-time bases, revision acceptance identity,
tenant-scoped knowledge ordering, position lifecycle, finalized watermarks, and
the domain-owned ``RevisionOrderingAuthority`` provider boundary.

Production persistence belongs to TRACE-BITEMP-2. This module MUST NOT import
vendor storage APIs or ``RuntimeEvent``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

_ACCEPTANCE_KEY_SUFFIX = re.compile(r"^[0-9a-f]{32}$")
_ACCEPTANCE_KEY_PREFIX = "rack_"


class CrossScopeKnowledgeOrderError(ValueError):
    """Raised when knowledge positions or watermarks from different scopes are mixed."""


class UnknownKnowledgeRevisionPositionError(ValueError):
    """Raised when a position was never allocated in the requested scope."""


class ValidTimeBoundKind(StrEnum):
    INSTANT = "instant"
    INTERVAL = "interval"


class SystemTimeBoundKind(StrEnum):
    INSTANT = "instant"
    INTERVAL = "interval"


class KnowledgeRevisionPositionLifecycle(StrEnum):
    """Canonical lifecycle of a knowledge-revision position.

    ``ALLOCATED`` is not ``ACCEPTED``. Readers MUST NOT treat allocated-only
    positions as accepted knowledge. ``ALLOCATED`` and ``UNRESOLVED`` both block
    watermark advancement. ``ACCEPTED`` and ``TERMINAL_NON_COMMITTED`` are
    finalized outcomes.
    """

    ALLOCATED = "allocated"
    ACCEPTED = "accepted"
    UNRESOLVED = "unresolved"
    TERMINAL_NON_COMMITTED = "terminal_non_committed"


def _require_aware_instant(value: object, label: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware datetime")
    return value


def _require_interval_bounds(*, start: datetime, end: datetime | None, label: str) -> None:
    if end is None:
        return
    aware_end = _require_aware_instant(end, f"{label}.end")
    if aware_end <= start:
        raise ValueError(f"{label} interval end must be strictly after start")


@dataclass(frozen=True, slots=True)
class ValidTimeBasis:
    """When the fact was valid/effective in the modeled domain.

    Instant: a single aware instant (``end`` MUST be ``None``).
    Interval: half-open ``[start, end)``; ``end is None`` means open-ended
    (no sentinel datetime). Supports backdating, retrospective correction,
    and future-effective ranges.
    """

    kind: ValidTimeBoundKind
    start: datetime
    end: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ValidTimeBoundKind):
            raise TypeError("ValidTimeBasis.kind must be ValidTimeBoundKind")
        object.__setattr__(self, "start", _require_aware_instant(self.start, "ValidTimeBasis.start"))
        if self.kind is ValidTimeBoundKind.INSTANT:
            if self.end is not None:
                raise ValueError("ValidTimeBasis instant must not set end")
            return
        _require_interval_bounds(start=self.start, end=self.end, label="ValidTimeBasis")

    @classmethod
    def instant(cls, at: datetime) -> ValidTimeBasis:
        return cls(kind=ValidTimeBoundKind.INSTANT, start=at, end=None)

    @classmethod
    def interval(cls, start: datetime, end: datetime | None) -> ValidTimeBasis:
        return cls(kind=ValidTimeBoundKind.INTERVAL, start=start, end=end)

    @property
    def is_open_ended(self) -> bool:
        return self.kind is ValidTimeBoundKind.INTERVAL and self.end is None


@dataclass(frozen=True, slots=True)
class SystemTimeBasis:
    """When Intergrax knew / recorded / accepted this revision.

    System time is a temporal axis. It is NOT revision ordering and MUST NOT
    be used alone to total-order concurrent corrections.
    Instant: recorded/accepted at ``start``.
    Interval: this belief was recorded during half-open ``[start, end)``;
    ``end is None`` means the belief is still current in system time.
    """

    kind: SystemTimeBoundKind
    start: datetime
    end: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SystemTimeBoundKind):
            raise TypeError("SystemTimeBasis.kind must be SystemTimeBoundKind")
        object.__setattr__(self, "start", _require_aware_instant(self.start, "SystemTimeBasis.start"))
        if self.kind is SystemTimeBoundKind.INSTANT:
            if self.end is not None:
                raise ValueError("SystemTimeBasis instant must not set end")
            return
        _require_interval_bounds(start=self.start, end=self.end, label="SystemTimeBasis")

    @classmethod
    def instant(cls, at: datetime) -> SystemTimeBasis:
        return cls(kind=SystemTimeBoundKind.INSTANT, start=at, end=None)

    @classmethod
    def interval(cls, start: datetime, end: datetime | None) -> SystemTimeBasis:
        return cls(kind=SystemTimeBoundKind.INTERVAL, start=start, end=end)

    @property
    def is_open_ended(self) -> bool:
        return self.kind is SystemTimeBoundKind.INTERVAL and self.end is None


@dataclass(frozen=True, slots=True)
class BitemporalKnowledgeBasis:
    """Bitemporal state = valid time + system time only.

    MUST NOT contain ``AsOfBoundary``, ``KnowledgeRevisionPosition``, or
    ``KnowledgeRevisionWatermark``.
    """

    valid_time: ValidTimeBasis
    system_time: SystemTimeBasis

    def __post_init__(self) -> None:
        if type(self.valid_time) is not ValidTimeBasis:
            raise TypeError("BitemporalKnowledgeBasis.valid_time must be ValidTimeBasis")
        if type(self.system_time) is not SystemTimeBasis:
            raise TypeError("BitemporalKnowledgeBasis.system_time must be SystemTimeBasis")


@dataclass(frozen=True, slots=True)
class KnowledgeOrderingScope:
    """Canonical TRACE-BITEMP-1 ordering scope: one tenant.

    This is the knowledge-ordering scope identity — not a platform-wide TenantId
    type and not interchangeable with ``RunId`` / ``EventId``.
    """

    tenant_id: str

    def __post_init__(self) -> None:
        if type(self.tenant_id) is not str:
            raise TypeError(
                f"KnowledgeOrderingScope.tenant_id must be str, got {type(self.tenant_id).__name__}"
            )
        if not self.tenant_id or self.tenant_id != self.tenant_id.strip():
            raise ValueError(
                "KnowledgeOrderingScope.tenant_id must be non-empty without surrounding whitespace"
            )


@dataclass(frozen=True, slots=True)
class RevisionAcceptanceKey:
    """Nominal stable identity for idempotent revision acceptance.

    Owner: the logical revision-acceptance operation (domain command / use-case)
    that calls ``RevisionOrderingAuthority.accept_revision``. The authority
    consumes the key; it does not mint a different retry identity.

    Scope: unique within ``KnowledgeOrderingScope``.

    Semantics: ``accept_revision(scope, acceptance_key=A) → K``; retry with the
    same ``A`` in the same scope returns the same accepted ``K``. Not a request
    timestamp, not ``EventId``, not ``RunId``.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", validate_revision_acceptance_key(self.value))


def validate_revision_acceptance_key(value: object) -> str:
    if isinstance(value, RevisionAcceptanceKey):
        return value.value
    if type(value) is not str:
        raise TypeError(
            f"RevisionAcceptanceKey must be str, got {type(value).__name__}"
        )
    if not value.startswith(_ACCEPTANCE_KEY_PREFIX):
        raise ValueError(f"RevisionAcceptanceKey must start with {_ACCEPTANCE_KEY_PREFIX!r}")
    suffix = value[len(_ACCEPTANCE_KEY_PREFIX) :]
    if not _ACCEPTANCE_KEY_SUFFIX.fullmatch(suffix):
        raise ValueError("RevisionAcceptanceKey suffix must match [0-9a-f]{32}")
    return value


def mint_revision_acceptance_key() -> RevisionAcceptanceKey:
    return RevisionAcceptanceKey(f"{_ACCEPTANCE_KEY_PREFIX}{uuid4().hex}")


@dataclass(frozen=True, slots=True)
class KnowledgeRevisionPosition:
    """Authoritative knowledge/revision acceptance position within one scope.

    Clock-independent positive integer. Not datetime, not
    ``ExecutionEventPosition``, not ``EventId``, not ``supersedes`` lineage.
    Positions from different scopes are not comparable as one sequence.
    """

    scope: KnowledgeOrderingScope
    value: int

    def __post_init__(self) -> None:
        if type(self.scope) is not KnowledgeOrderingScope:
            raise TypeError("KnowledgeRevisionPosition.scope must be KnowledgeOrderingScope")
        if type(self.value) is not int or isinstance(self.value, bool) or self.value < 1:
            raise ValueError("KnowledgeRevisionPosition.value must be a positive int >= 1")

    def precedes(self, other: KnowledgeRevisionPosition) -> bool:
        _require_same_scope(self.scope, other.scope, "KnowledgeRevisionPosition")
        return self.value < other.value


@dataclass(frozen=True, slots=True)
class KnowledgeRevisionWatermark:
    """Highest authoritative position K such that every position <= K in this
    scope has a durable finalized outcome (ACCEPTED or TERMINAL_NON_COMMITTED).

    ``finalized_through_value == 0`` means the empty prefix (nothing finalized).
    This is not highest-allocated, not highest-observed, and not highest-accepted
    while skipping unresolved lower positions.
    """

    scope: KnowledgeOrderingScope
    finalized_through_value: int

    def __post_init__(self) -> None:
        if type(self.scope) is not KnowledgeOrderingScope:
            raise TypeError("KnowledgeRevisionWatermark.scope must be KnowledgeOrderingScope")
        if (
            type(self.finalized_through_value) is not int
            or isinstance(self.finalized_through_value, bool)
            or self.finalized_through_value < 0
        ):
            raise ValueError(
                "KnowledgeRevisionWatermark.finalized_through_value must be int >= 0"
            )

    @property
    def is_empty(self) -> bool:
        return self.finalized_through_value == 0

    @property
    def finalized_through(self) -> KnowledgeRevisionPosition | None:
        if self.finalized_through_value == 0:
            return None
        return KnowledgeRevisionPosition(scope=self.scope, value=self.finalized_through_value)

    def includes(self, position: KnowledgeRevisionPosition) -> bool:
        _require_same_scope(self.scope, position.scope, "KnowledgeRevisionWatermark")
        return position.value <= self.finalized_through_value


@dataclass(frozen=True, slots=True)
class KnowledgeRevisionWatermarkSet:
    """Cross-scope historical query result.

    Not a global K. Callers MUST NOT treat tenant K12 and tenant K20 as one
    sequence. Duplicate scopes are forbidden.
    """

    watermarks: tuple[KnowledgeRevisionWatermark, ...]

    def __post_init__(self) -> None:
        seen: set[KnowledgeOrderingScope] = set()
        for watermark in self.watermarks:
            if type(watermark) is not KnowledgeRevisionWatermark:
                raise TypeError(
                    "KnowledgeRevisionWatermarkSet entries must be KnowledgeRevisionWatermark"
                )
            if watermark.scope in seen:
                raise ValueError("KnowledgeRevisionWatermarkSet must not contain duplicate scopes")
            seen.add(watermark.scope)


@dataclass(frozen=True, slots=True)
class KnowledgeRevisionPositionRecord:
    position: KnowledgeRevisionPosition
    lifecycle: KnowledgeRevisionPositionLifecycle

    def __post_init__(self) -> None:
        if type(self.position) is not KnowledgeRevisionPosition:
            raise TypeError(
                "KnowledgeRevisionPositionRecord.position must be KnowledgeRevisionPosition"
            )
        if not isinstance(self.lifecycle, KnowledgeRevisionPositionLifecycle):
            raise TypeError(
                "KnowledgeRevisionPositionRecord.lifecycle must be KnowledgeRevisionPositionLifecycle"
            )


@dataclass(frozen=True, slots=True)
class KnowledgeRevisionAcceptance:
    """Idempotent accept result: stable key bound to authoritative position K."""

    acceptance_key: RevisionAcceptanceKey
    position: KnowledgeRevisionPosition

    def __post_init__(self) -> None:
        if type(self.acceptance_key) is not RevisionAcceptanceKey:
            raise TypeError(
                "KnowledgeRevisionAcceptance.acceptance_key must be RevisionAcceptanceKey"
            )
        if type(self.position) is not KnowledgeRevisionPosition:
            raise TypeError(
                "KnowledgeRevisionAcceptance.position must be KnowledgeRevisionPosition"
            )


def lifecycle_is_finalized(lifecycle: KnowledgeRevisionPositionLifecycle) -> bool:
    return lifecycle in (
        KnowledgeRevisionPositionLifecycle.ACCEPTED,
        KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED,
    )


def lifecycle_blocks_watermark(lifecycle: KnowledgeRevisionPositionLifecycle) -> bool:
    return lifecycle in (
        KnowledgeRevisionPositionLifecycle.ALLOCATED,
        KnowledgeRevisionPositionLifecycle.UNRESOLVED,
    )


def compute_finalized_watermark(
    *,
    scope: KnowledgeOrderingScope,
    records: tuple[KnowledgeRevisionPositionRecord, ...],
) -> KnowledgeRevisionWatermark:
    """Pure finalized-contiguous watermark rule.

    Every position in ``1..max`` MUST be present and classifiable. Missing keys
    are forbidden (no invisible gaps). ``ALLOCATED`` / ``UNRESOLVED`` stop
    advancement immediately before that position. ``TERMINAL_NON_COMMITTED``
    does not block advancement.
    """

    if type(scope) is not KnowledgeOrderingScope:
        raise TypeError("scope must be KnowledgeOrderingScope")
    if type(records) is not tuple:
        raise TypeError("records must be a tuple of KnowledgeRevisionPositionRecord")
    if not records:
        return KnowledgeRevisionWatermark(scope=scope, finalized_through_value=0)

    by_value: dict[int, KnowledgeRevisionPositionLifecycle] = {}
    for record in records:
        if type(record) is not KnowledgeRevisionPositionRecord:
            raise TypeError("records must contain KnowledgeRevisionPositionRecord only")
        if record.position.scope != scope:
            raise CrossScopeKnowledgeOrderError(
                "position scope does not match watermark computation scope"
            )
        value = record.position.value
        if value in by_value:
            raise ValueError(f"duplicate knowledge position {value} in scope")
        by_value[value] = record.lifecycle

    max_value = max(by_value)
    missing = [value for value in range(1, max_value + 1) if value not in by_value]
    if missing:
        raise ValueError(
            f"knowledge position map must be contiguous from 1; missing {missing!r}"
        )

    finalized_through = 0
    for value in range(1, max_value + 1):
        lifecycle = by_value[value]
        if lifecycle_blocks_watermark(lifecycle):
            break
        if not lifecycle_is_finalized(lifecycle):
            raise ValueError(f"unclassifiable knowledge position lifecycle {lifecycle}")
        finalized_through = value
    return KnowledgeRevisionWatermark(scope=scope, finalized_through_value=finalized_through)


class RevisionOrderingAuthority(ABC):
    """Observability / bitemporal domain-owned revision ordering contract.

    Applications consume this contract. Applications MUST NOT implement custom
    ordering semantics. Host/deployment DI selects the provider. Public core
    types are vendor-neutral.

    Canonical first-party production strategy (TRACE-BITEMP-1): one durable
    transactional boundary coordinating acceptance identity, position allocation,
    durable acceptance, and lifecycle/finality. Physical store belongs to
    TRACE-BITEMP-2.
    """

    @abstractmethod
    def accept_revision(
        self,
        *,
        scope: KnowledgeOrderingScope,
        acceptance_key: RevisionAcceptanceKey,
    ) -> KnowledgeRevisionAcceptance:
        """Idempotently accept a logical revision in ``scope``.

        Retry with the same ``acceptance_key`` MUST return the same ``position``.
        """

    @abstractmethod
    def position_lifecycle(
        self,
        position: KnowledgeRevisionPosition,
    ) -> KnowledgeRevisionPositionLifecycle:
        """Classify a known position. Never returns a missing/None state.

        Unknown (never allocated) positions raise ``UnknownKnowledgeRevisionPositionError``.
        """

    @abstractmethod
    def watermark(self, scope: KnowledgeOrderingScope) -> KnowledgeRevisionWatermark:
        """Finalized contiguous watermark for ``scope``."""

    @abstractmethod
    def records_through(
        self,
        watermark: KnowledgeRevisionWatermark,
    ) -> tuple[KnowledgeRevisionPositionRecord, ...]:
        """Every position ``<= watermark`` with a finalized lifecycle, lowest first."""

    @abstractmethod
    def unresolved_positions(
        self,
        scope: KnowledgeOrderingScope,
    ) -> tuple[KnowledgeRevisionPosition, ...]:
        """Positions that currently block watermark advancement in ``scope``."""


def _require_same_scope(
    left: KnowledgeOrderingScope,
    right: KnowledgeOrderingScope,
    label: str,
) -> None:
    if left != right:
        raise CrossScopeKnowledgeOrderError(
            f"{label} forbids mixing distinct knowledge ordering scopes"
        )
