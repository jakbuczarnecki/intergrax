# PLAN T-IGX-066-B — Wave 06 OD-042 (CONFIG-X neutral globals wariant W1)

Evidence pin (read): `da5ca5b53fbe510e655b12bc91ab4e6af1cef6e4`  
EQO finding: **EQOA-A4-042** (Poland-specific `GlobalSettings` defaults).

## 1. Problem

`intergrax/globals/settings.py` sets import-time defaults `pl`, `pl-PL`, `Europe/Warsaw`. No tenant-neutral platform baseline; OD-EQO-09 does **not** authorize this change.

## 2. Goal (variant W1)

- Neutral defaults: `en` / `en-US` / `UTC` (BCP47 + IANA).
- **Persistent storage** remains UTC everywhere it already is; this ticket only changes SSOT defaults and documents that presentation may use tenant/host override.
- Environment variables `INTERGRAX_DEFAULT_*` continue to override defaults (backward compatible for operators using Poland env).
- No repo-wide `os.getenv` migration; no changes to vendor-knowledge tests that explicitly use `Europe/Warsaw` for DST/calendar scenarios.

## 3. Exact write-set

```text
intergrax/globals/settings.py
tests/unit/globals/test_global_settings_neutral_defaults.py
tests/unit/globals/test_global_settings_env_override_isolation.py
governance/tickets/T-IGX-066-B.md
governance/tickets/plans/PLAN_T-IGX-066-B.md
```

**Forbidden:** `HostProfile` contract edits; mass test rewrites; CONFIG-X parent re-close claims; FRZ promotion.

## 4. Implementation rules

1. Change class-field defaults only (keep `os.getenv(..., new_default)` pattern).
2. Docstring on `GlobalSettings`: storage uses UTC; display/locale from tenant or env override.
3. Unit tests must not mutate process env globally without `monkeypatch` / isolated subprocess pattern.

## 5. Dependencies

- Operator `Approved:` OD-042 + this plan digest.
- Serialize after **T-IGX-066-A** if both approved same session (no file overlap).

## 6. Acceptance tests

```text
uv run pytest tests/unit/globals/test_global_settings_neutral_defaults.py tests/unit/globals/test_global_settings_env_override_isolation.py -q
uv run pytest tests/qualification/config_x/test_config_x_qualification_gates.py tests/qualification/config_x/test_config_x_r1_r1_semantic_constant_resolution.py -q --tb=line
```

## 7. Negative tests

| Test | Intent |
| --- | --- |
| `test_global_settings_env_override_isolation` | `INTERGRAX_DEFAULT_LOCALE=pl-PL` wins over neutral default |
| `test_global_settings_env_override_isolation` | Two logical tenants simulated via sequential env patches — no bleed |
| Fresh `GlobalSettings()` after env unset | Returns neutral defaults |

## 8. Compatibility

- Deployments with explicit env: **no behavior change**.
- Deployments without env: language/locale shift Poland → US English; operators must set env if they require Poland.
- CONFIG-X historical PASS @ `fac9a43f` not revoked; local requalification via subset gates above.

## 9. Rollback

Revert `settings.py` and unit tests; defaults return to Poland. No data migration required.

## 10. Independent review

- `igx-reviewer`: verify write-set only; **REJECT** if unrelated modules or HostProfile touched.
- Not G6.

## 11. Publication

Same as T-IGX-066-A with `Ref: governance/tickets/T-IGX-066-B.md`.