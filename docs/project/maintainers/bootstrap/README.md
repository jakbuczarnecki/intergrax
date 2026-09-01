# Cursor session bootstrap (paste files)

**Doc map:** [`../DOCUMENTATION_MAP.md`](../../technical/DOCUMENTATION_MAP.md) - bootstrap vs audit vs audit_results roles.
**F3 session rule:** `ONE DOMAIN = ONE NEW CHAT` - see [`../../technical/guides/CURSOR_TOKEN_SETUP.md`](../../technical/guides/CURSOR_TOKEN_SETUP.md).
**I1/O1:** always-on `.cursor/rules/intergrax-token-budget.mdc`. **F2 setup:** root `AGENTS.md` is a Cursor auto-load stub; full reference in `docs/project/technical/guides/AGENT_INSTRUCTIONS.md`. Do not delete stub via Settings trash icon.

Copy **entire file** into a **new** Cursor agent chat as the first message.

| File | Mode | Scope |
|------|------|-------|
| [`micro_implement.txt`](micro_implement.txt) | MICRO - bounded implementation | One small task; explicit read/edit scope; max 8 reads; no semantic search |
| [`07_ci_preflight.txt`](07_ci_preflight.txt) | **CI - preflight before push/merge** | Run local parity with `.github/workflows/unit-tests.yml` (`--profile all`) |
| [`hep_step.txt`](hep_step.txt) | **HEP - one plan step** | Phase HEP C13–C16 / EVID-* rows - edit `STEP=` / `SCOPE=` |

**Mode MICRO:** Use [`micro_implement.txt`](micro_implement.txt) as the default bootstrap for ordinary small implementation tasks outside HEP/EVID and outside CI/test/checker hotfixes. Use a new Cursor chat, list exact files and line ranges, and reject any run that reads outside scope.

**Platform audit (no code):** Follow [`docs/audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md). Do not use bootstrap paste files for audits.

**Remediation from audit:** Follow [`docs/audit_results/AUDIT_REMEDIATION_PROTOCOL.md`](../../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).

**Layer closeout (LCM):** [`LAYER_COMPLETION_MODE.md`](../../technical/guides/LAYER_COMPLETION_MODE.md) - deep single-layer maturity workflow.
