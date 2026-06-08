# `ops.findings_review`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Security findings triage with scan, summarize, and notify.

## How it works

security.summarize_findings + security.scan + notify.send.

## How to use

ops_skill_profile(); restrict to HIGH risk trusted hosts.

## What you get

Findings review loop distinct from security_audit scan-only path.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `security.summarize_findings` | Summarize scan findings |
| `security.scan` | Run security scan |
| `notify.send` | Alert owners |

## Related skills

- `ops.security_audit`
- `ops.incident_dispatch`
