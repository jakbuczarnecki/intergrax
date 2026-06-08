# `platform.secrets_flags`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Runtime secrets and feature-flag evaluation.

## How it works

platform.get_secret + evaluate_feature_flag bindings.

## How to use

Restrict to HIGH risk agents on trusted hosts.

## What you get

Governed secrets/flags without env leakage in agents.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `platform.get_secret` | Fetch secret by key |
| `platform.evaluate_feature_flag` | Evaluate feature flag |

## Related skills

- `platform.concierge`
- `ops.security_audit`
