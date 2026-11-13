# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recurrence as one-shot occurrences on the existing scheduler ledger (ADR-PDC-07).

A ``RecurrenceRule`` (fixed interval or 5-field cron, IANA time zone, misfire policy)
is a pure schedule description. It never fires anything and owns no loop: after each
occurrence the existing ``LongRunningScheduler`` asks ``RecurrenceMaterializer`` to
persist the NEXT occurrence as an ordinary one-shot ``ScheduledResume`` through the
existing ``ScheduledResumePersistence`` / ``SchedulerLedger`` store. Every occurrence
is therefore claimed, fenced, and admitted for execution exactly like any other
scheduled resume (no second scheduler, no execution-engine change).

Semantics (all instants are stored and compared in UTC):

* ``interval_seconds`` is elapsed time anchored at ``start_utc`` and is DST-immune.
* ``cron`` is evaluated on the wall clock of ``timezone``:
  - a fixed-hour rule whose local time falls in a spring-forward gap fires ONCE at the
    first valid instant after the gap (never skipped, never doubled);
  - a fixed-hour rule whose local time is repeated by a fall-back fires ONCE, at the
    first occurrence of that wall time;
  - a wildcard-hour rule (``*`` or ``*/n`` in the hour field) follows elapsed time:
    repeated hours fire for both passes, missing hours do not fire.
* Misfire (the scheduler was down and whole occurrences were missed) is resolved per
  rule: ``skip`` drops them, ``fire_once`` runs the latest missed one, ``catch_up(n)``
  runs the most recent ``n`` missed ones. Scans are bounded by ``MAX_MISFIRE_SCAN``;
  beyond it the policy degrades to ``skip`` (fail-safe).

Occurrence identity is deterministic (``rule_id`` + occurrence instant), so two
processes materializing the same occurrence converge on one row: the ledger rejects
the duplicate ``schedule_id``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from functools import lru_cache
from itertools import count
from typing import Any, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
)
from intergrax.runtime.long_running.scheduled_resume_metadata import (
    validate_scheduled_resume_metadata,
)
from intergrax.runtime.long_running.scheduler_claim import (
    ScheduledResumeScheduleConflictError,
)

__all__ = [
    "MAX_MISFIRE_SCAN",
    "RECURRENCE_OCCURRENCE_METADATA_KEY",
    "RECURRENCE_RULE_METADATA_KEY",
    "MisfirePolicy",
    "MisfirePolicyKind",
    "RecurrenceError",
    "RecurrenceMaterializer",
    "RecurrenceRule",
    "occurrence_schedule_id",
    "plan_next_occurrences",
]

logger = logging.getLogger(__name__)

MAX_MISFIRE_SCAN = 10_000
RECURRENCE_RULE_METADATA_KEY = "recurrence_rule"
RECURRENCE_OCCURRENCE_METADATA_KEY = "recurrence_occurrence_utc"

_MAX_IDLE_SCAN_DAYS = 366 * 9  # longest legitimate gap between occurrences (leap-day rules)
_RULE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MONTH_NAMES = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_DOW_NAMES = {name: index for index, name in enumerate(("sun", "mon", "tue", "wed", "thu", "fri", "sat"))}


class RecurrenceError(ValueError):
    """Recurrence rule or persisted occurrence is invalid (fail-closed)."""


# --------------------------------------------------------------------------- time helpers


def _as_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RecurrenceError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc_iso(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise RecurrenceError(f"{label} must be an ISO-8601 timestamp") from exc
    return _as_utc(parsed, label)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


# --------------------------------------------------------------------------- cron


class _CronSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]  # 0 = Sunday
    day_of_month_wildcard: bool
    day_of_week_wildcard: bool
    hour_wildcard: bool


def _field_value(token: str, names: Mapping[str, int], label: str) -> int:
    lowered = token.lower()
    if lowered in names:
        return names[lowered]
    if not token.isdigit():
        raise RecurrenceError(f"cron {label}: invalid value {token!r}")
    return int(token)


def _parse_cron_field(
    text: str,
    low: int,
    high: int,
    label: str,
    names: Mapping[str, int] | None = None,
) -> frozenset[int]:
    resolved_names = names or {}
    values: set[int] = set()
    for part in text.split(","):
        if not part:
            raise RecurrenceError(f"cron {label}: empty list element")
        step = 1
        base = part
        has_step = "/" in part
        if has_step:
            base, _, step_text = part.partition("/")
            if not step_text.isdigit() or int(step_text) < 1:
                raise RecurrenceError(f"cron {label}: invalid step {step_text!r}")
            step = int(step_text)
        if base == "*":
            start, end = low, high
        elif "-" in base:
            left, _, right = base.partition("-")
            start = _field_value(left, resolved_names, label)
            end = _field_value(right, resolved_names, label)
        else:
            start = _field_value(base, resolved_names, label)
            end = high if has_step else start
        if not (low <= start <= end <= high):
            raise RecurrenceError(f"cron {label}: {part!r} outside {low}-{high}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


@lru_cache(maxsize=256)
def _compile_cron(expression: str) -> _CronSpec:
    fields = expression.split()
    if len(fields) != 5:
        raise RecurrenceError("cron expression must have exactly 5 fields")
    minute_f, hour_f, dom_f, month_f, dow_f = fields
    raw_dow = _parse_cron_field(dow_f, 0, 7, "day-of-week", _DOW_NAMES)
    return _CronSpec(
        minutes=_parse_cron_field(minute_f, 0, 59, "minute"),
        hours=_parse_cron_field(hour_f, 0, 23, "hour"),
        days_of_month=_parse_cron_field(dom_f, 1, 31, "day-of-month"),
        months=_parse_cron_field(month_f, 1, 12, "month", _MONTH_NAMES),
        days_of_week=frozenset(0 if value == 7 else value for value in raw_dow),
        day_of_month_wildcard=dom_f.startswith("*"),
        day_of_week_wildcard=dow_f.startswith("*"),
        hour_wildcard=hour_f.startswith("*"),
    )


def _load_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise RecurrenceError(f"unknown IANA time zone {name!r}") from exc


# --------------------------------------------------------------------------- misfire


class MisfirePolicyKind(StrEnum):
    SKIP = "skip"
    FIRE_ONCE = "fire_once"
    CATCH_UP = "catch_up"


class MisfirePolicy(BaseModel):
    """What to do with occurrences missed while the scheduler was not running."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: MisfirePolicyKind = MisfirePolicyKind.SKIP
    max_catch_up: int = 0

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.kind is MisfirePolicyKind.CATCH_UP:
            if self.max_catch_up < 1:
                raise ValueError("catch_up requires max_catch_up >= 1")
        elif self.max_catch_up != 0:
            raise ValueError(f"{self.kind.value} does not take max_catch_up")
        return self

    @classmethod
    def skip(cls) -> MisfirePolicy:
        return cls(kind=MisfirePolicyKind.SKIP)

    @classmethod
    def fire_once(cls) -> MisfirePolicy:
        return cls(kind=MisfirePolicyKind.FIRE_ONCE)

    @classmethod
    def catch_up(cls, n: int) -> MisfirePolicy:
        return cls(kind=MisfirePolicyKind.CATCH_UP, max_catch_up=n)


# --------------------------------------------------------------------------- rule


class RecurrenceRule(BaseModel):
    """Pure schedule description that targets one paused-task resume."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    tenant_id: str
    task_id: str
    resume_token: str
    interval_seconds: int | None = None
    cron: str | None = None
    timezone: str = "UTC"
    start_utc: str
    end_utc: str | None = None
    misfire: MisfirePolicy = Field(default_factory=MisfirePolicy.skip)
    resume_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_rule(self) -> Self:
        if not _RULE_ID_PATTERN.fullmatch(self.rule_id):
            raise ValueError("rule_id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
        if (self.interval_seconds is None) == (self.cron is None):
            raise ValueError("exactly one of interval_seconds or cron is required")
        if self.interval_seconds is not None and self.interval_seconds < 1:
            raise ValueError("interval_seconds must be >= 1")
        if self.cron is not None:
            _compile_cron(self.cron)
        _load_zone(self.timezone)
        start = _parse_utc_iso(self.start_utc, "start_utc")
        if self.end_utc is not None and _parse_utc_iso(self.end_utc, "end_utc") < start:
            raise ValueError("end_utc must not precede start_utc")
        validate_scheduled_resume_metadata(self.resume_metadata)
        return self

    # -- derived views

    @property
    def start(self) -> datetime:
        return _parse_utc_iso(self.start_utc, "start_utc")

    @property
    def end(self) -> datetime | None:
        return _parse_utc_iso(self.end_utc, "end_utc") if self.end_utc else None

    def first_occurrence(self) -> datetime | None:
        """First occurrence at or after ``start_utc`` (None when past ``end_utc``)."""
        return self._first(self.start - timedelta(microseconds=1))

    def next_after(self, after: datetime) -> datetime | None:
        """Next occurrence strictly after ``after`` (UTC), or None past ``end_utc``."""
        return self._first(_as_utc(after, "after"))

    def occurrences_between(self, after: datetime, until: datetime, limit: int) -> list[datetime]:
        """Occurrences in ``(after, until]`` in order, at most ``limit`` of them."""
        horizon = _as_utc(until, "until")
        found: list[datetime] = []
        for when in self._iter_after(_as_utc(after, "after")):
            if when > horizon or len(found) >= limit:
                break
            found.append(when)
        return found

    # -- internals

    def _first(self, after_utc: datetime) -> datetime | None:
        return next(self._iter_after(after_utc), None)

    def _iter_after(self, after_utc: datetime) -> Iterator[datetime]:
        """Strictly increasing UTC occurrences after ``after_utc`` up to ``end_utc``."""
        end = self.end
        floor = max(after_utc, self.start - timedelta(microseconds=1))
        source = self._iter_interval(floor) if self.interval_seconds is not None else self._iter_cron(floor)
        for when in source:
            if end is not None and when > end:
                return
            yield when

    def _iter_interval(self, floor: datetime) -> Iterator[datetime]:
        if self.interval_seconds is None:
            raise RecurrenceError("not an interval rule")
        step = timedelta(seconds=self.interval_seconds)
        start = self.start
        first_index = 0 if floor < start else (floor - start) // step + 1
        for index in count(first_index):
            yield start + index * step

    def _iter_cron(self, floor: datetime) -> Iterator[datetime]:
        if self.cron is None:
            raise RecurrenceError("not a cron rule")
        spec = _compile_cron(self.cron)
        zone = _load_zone(self.timezone)
        day = floor.astimezone(zone).date() - timedelta(days=1)
        last = floor
        idle_days = 0
        while idle_days < _MAX_IDLE_SCAN_DAYS:
            idle_days += 1
            current, day = day, day + timedelta(days=1)
            if not self._day_matches(spec, current):
                continue
            for when in sorted(self._day_instants(spec, zone, current)):
                if when > last:
                    last = when
                    idle_days = 0
                    yield when

    @staticmethod
    def _day_matches(spec: _CronSpec, day: date) -> bool:
        if day.month not in spec.months:
            return False
        day_of_month_ok = day.day in spec.days_of_month
        day_of_week_ok = (day.weekday() + 1) % 7 in spec.days_of_week
        if spec.day_of_month_wildcard or spec.day_of_week_wildcard:
            return day_of_month_ok and day_of_week_ok
        return day_of_month_ok or day_of_week_ok

    @staticmethod
    def _day_instants(spec: _CronSpec, zone: ZoneInfo, day: date) -> set[datetime]:
        instants: set[datetime] = set()
        for hour in spec.hours:
            for minute in spec.minutes:
                wall = datetime(day.year, day.month, day.day, hour, minute)
                first = wall.replace(tzinfo=zone, fold=0).astimezone(UTC)
                exists = first.astimezone(zone).replace(tzinfo=None) == wall
                if not exists:
                    # Spring-forward gap: fixed-hour rules fire once, shifted past the
                    # gap; wildcard-hour rules simply have no such slot.
                    if not spec.hour_wildcard:
                        instants.add(first)
                    continue
                instants.add(first)
                if spec.hour_wildcard:
                    # Fall-back overlap: wildcard-hour rules run on elapsed time.
                    instants.add(wall.replace(tzinfo=zone, fold=1).astimezone(UTC))
        return instants


# --------------------------------------------------------------------------- planning


def occurrence_schedule_id(rule_id: str, occurrence: datetime) -> str:
    """Deterministic global identity of one occurrence (rule + UTC instant)."""
    stamp = _as_utc(occurrence, "occurrence").strftime("%Y%m%dT%H%M%SZ")
    return f"rec_{rule_id}_{stamp}"


def plan_next_occurrences(rule: RecurrenceRule, after: datetime, now: datetime) -> list[datetime]:
    """Occurrences to materialize after ``after`` fired, given the current ``now``."""
    current = _as_utc(now, "now")
    nxt = rule.next_after(after)
    if nxt is None:
        return []
    if nxt > current:
        return [nxt]

    # At least one whole occurrence was missed: apply the misfire policy.
    policy = rule.misfire
    if policy.kind is not MisfirePolicyKind.SKIP:
        missed = rule.occurrences_between(after, current, MAX_MISFIRE_SCAN + 1)
        if len(missed) <= MAX_MISFIRE_SCAN:
            if policy.kind is MisfirePolicyKind.FIRE_ONCE:
                return [missed[-1]]
            return missed[-policy.max_catch_up :]
        logger.warning(
            "recurrence %s missed more than %d occurrences; degrading to skip",
            rule.rule_id,
            MAX_MISFIRE_SCAN,
        )
    future = rule.next_after(current)
    return [future] if future is not None else []


# --------------------------------------------------------------------------- materializer


class RecurrenceMaterializer:
    """Persists recurrence occurrences as one-shot resumes on the existing ledger."""

    def __init__(self, schedule_store: ScheduledResumePersistence) -> None:
        self._schedule_store = schedule_store

    def start(self, rule: RecurrenceRule) -> ScheduledResume | None:
        """Materialize the first occurrence; None when it already exists or none remain."""
        first = rule.first_occurrence()
        if first is None:
            return None
        return self._materialize(rule, first)

    def materialize_next(self, entry: ScheduledResume, *, now: datetime) -> list[ScheduledResume]:
        """Materialize the occurrence(s) that follow a fired recurring ``entry``.

        Entries that do not belong to a recurrence return ``[]``. Already-existing
        occurrences (another process won the race) are skipped silently.
        """
        raw_rule = entry.resume_metadata.get(RECURRENCE_RULE_METADATA_KEY)
        if raw_rule is None:
            return []
        rule = self._rule_for_entry(entry, raw_rule)
        raw_occurrence = entry.resume_metadata.get(RECURRENCE_OCCURRENCE_METADATA_KEY, entry.run_at_utc)
        fired = _parse_utc_iso(str(raw_occurrence), "recurrence occurrence")
        created: list[ScheduledResume] = []
        for occurrence in plan_next_occurrences(rule, fired, now):
            row = self._materialize(rule, occurrence)
            if row is not None:
                created.append(row)
        return created

    @staticmethod
    def _rule_for_entry(entry: ScheduledResume, raw_rule: object) -> RecurrenceRule:
        if not isinstance(raw_rule, dict):
            raise RecurrenceError("persisted recurrence rule is not an object")
        rule = RecurrenceRule.model_validate(raw_rule)
        if (
            rule.tenant_id != entry.tenant_id
            or rule.task_id != entry.task_id
            or rule.resume_token != entry.resume_token
        ):
            raise RecurrenceError("persisted recurrence rule does not match its occurrence")
        return rule

    def _materialize(self, rule: RecurrenceRule, occurrence: datetime) -> ScheduledResume | None:
        when = _iso(occurrence)
        metadata: dict[str, Any] = dict(rule.resume_metadata)
        metadata[RECURRENCE_RULE_METADATA_KEY] = rule.model_dump(mode="json")
        metadata[RECURRENCE_OCCURRENCE_METADATA_KEY] = when
        entry = ScheduledResume(
            schedule_id=occurrence_schedule_id(rule.rule_id, occurrence),
            task_id=rule.task_id,
            tenant_id=rule.tenant_id,
            resume_token=rule.resume_token,
            run_at_utc=when,
            resume_metadata=metadata,
        )
        try:
            return self._schedule_store.schedule(entry)
        except ScheduledResumeScheduleConflictError:
            return None
