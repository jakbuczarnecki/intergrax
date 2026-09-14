# © Artur Czarnecki. All rights reserved.

"""Runbook section contract for EE-B4-C production runbooks."""

from __future__ import annotations

from enum import StrEnum


class RunbookSectionId(StrEnum):
    TITLE = "Title"
    TRIGGER = "Trigger"
    SEVERITY = "Severity"
    SCOPE = "Scope"
    SYMPTOMS = "Symptoms"
    CANONICAL_SIGNALS = "Canonical signals"
    IMMEDIATE_SAFE_ACTION = "Immediate safe action"
    DO_NOT = "Do NOT"
    DIAGNOSIS = "Diagnosis"
    RECOVERY_PATH = "Recovery path"
    VERIFICATION = "Verification"
    ESCALATION = "Escalation"
    POST_INCIDENT_EVIDENCE = "Post-incident evidence"


RUNBOOK_REQUIRED_SECTIONS: tuple[RunbookSectionId, ...] = tuple(RunbookSectionId)
