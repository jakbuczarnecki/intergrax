# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q2 — one-command real tool-selection qualification."""

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
$env:LKW_BASE_URL = if ($env:LKW_BASE_URL) { $env:LKW_BASE_URL } else { "http://localhost:8021" }
$env:LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY = if ($env:LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY) {
    $env:LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY
} else {
    "ue-11g-c1-certification-secret"
}
uv run python -m tests.system.functional_diagnostics_q2.runner
