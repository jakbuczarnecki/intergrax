# © Artur Czarnecki. All rights reserved.

"""External dependency preflight classification for diagnostic qualifications."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    DiagnosticQualificationDependencyStatus,
    ExternalDependencyState,
    QualificationFamily,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _mongodb_available() -> bool:
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    if not uri.strip():
        return False
    try:
        from pymongo import MongoClient
    except ImportError:
        return False
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return True
    except Exception:
        return False


def _http_reachable(url: str, timeout_seconds: float = 2.0) -> bool:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _lkw_ready() -> ExternalDependencyState:
    base_url = os.environ.get("LKW_BASE_URL", "http://localhost:8021").rstrip("/")
    if not _http_reachable(f"{base_url}/health"):
        return ExternalDependencyState.BLOCKED_SERVICE_UNAVAILABLE
    api_key = os.environ.get("LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY", "")
    if not api_key:
        return ExternalDependencyState.BLOCKED_MISSING_CREDENTIAL
    return ExternalDependencyState.READY


def _mongodb_ready() -> ExternalDependencyState:
    if _mongodb_available():
        return ExternalDependencyState.READY
    uri = os.environ.get("MONGODB_URI", "")
    if not uri:
        return ExternalDependencyState.BLOCKED_MISSING_CREDENTIAL
    return ExternalDependencyState.BLOCKED_SERVICE_UNAVAILABLE


def _tavily_ready() -> ExternalDependencyState:
    if not os.environ.get("TAVILY_API_KEY", "").strip():
        return ExternalDependencyState.BLOCKED_MISSING_CREDENTIAL
    return ExternalDependencyState.READY


def _model_routing_ready() -> ExternalDependencyState:
    if os.environ.get("DIAG_FUNCTIONAL_Q4_SKIP", "").strip().lower() in {"1", "true", "yes"}:
        return ExternalDependencyState.NOT_EXECUTED
    return ExternalDependencyState.READY


def classify_external_dependencies() -> tuple[DiagnosticQualificationDependencyStatus, ...]:
    lkw_state = _lkw_ready()
    mongo_state = _mongodb_ready()
    tavily_state = _tavily_ready()
    model_state = _model_routing_ready()
    return (
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.Q1,
            state=lkw_state if mongo_state is ExternalDependencyState.READY else mongo_state,
            required_services=("lkw", "qdrant", "ollama", "mongodb"),
            note="Real RAG/C1 path via LKW stack",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.Q2,
            state=lkw_state,
            required_services=("lkw", "mongodb"),
            note="Real tool selection via LKW",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.Q3,
            state=tavily_state,
            required_services=("tavily", "web_search"),
            note="Real web search qualification",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.Q4,
            state=model_state,
            required_services=("model_routing",),
            note="Real model routing qualification",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.Q5,
            state=ExternalDependencyState.READY,
            required_services=("in_process_plugins",),
            note="Cross-domain in-process plugin qualification",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.D1,
            state=mongo_state,
            required_services=("mongodb",),
            note="Durable DocumentStore durability proof",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.S1,
            state=mongo_state,
            required_services=("mongodb",),
            note="Production scale structural qualification",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.R1,
            state=mongo_state,
            required_services=("mongodb",),
            note="Bounded read-path qualification",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.R1_R1,
            state=mongo_state,
            required_services=("mongodb",),
            note="Projection migration recovery",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.R1_R2,
            state=mongo_state,
            required_services=("mongodb",),
            note="Append crash recovery",
        ),
        DiagnosticQualificationDependencyStatus(
            family=QualificationFamily.R1_R3,
            state=mongo_state,
            required_services=("mongodb",),
            note="Active writer fail-closed safety",
        ),
    )
