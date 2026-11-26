---
trigger: model_decision
description: Environment configuration, secrets handling, vendor integration qualification, and C10/C15 Windows freeze guardrails.
---

# Environment Config & Vendor Qualification (Antigravity adapter)

Canonical rule (SSOT): `agent-env/rules/intergrax-env-vendor-qualification.md`

- Pass environment values only via environment variables and approved typed config contracts; never hardcode credentials, tokens, or prod endpoints in source, tests, or logs.
- C10/C15 operator freeze: preserve original Windows execution environment; no Docker migration, no new vendor integrations, no production configuration changes.
- Read `agent-env/rules/intergrax-env-vendor-qualification.md` before changing configuration, integrations, or qualification setups.
