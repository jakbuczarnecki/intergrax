# CTRL-X wide pyright provenance

Mechanical SSOT:

- File list: `tests/qualification/control_planes/ctrl_x/wide_pyright_scope.py`
- Grouped diagnostics: `tests/qualification/control_planes/ctrl_x/wide_pyright_provenance.py`
- Semantic-boundary modules (0-error gate): `tests/qualification/control_planes/ctrl_x/typing_scope.py`

## Command (identical at A / B / C)

```bash
uv run pyright --outputjson $(CTRL_X_WIDE_PYRIGHT_FILE_PATHS from wide_pyright_scope.py)
```

Same pyright (project dev extra), Python 3.12, default pyproject config, explicit file paths only.

## HEAD pins

| Label | SHA |
|---|---|
| A pre-CTRL-X baseline | `3e574ea35ee2e3f9e9adac4088cd51ae89ed469c` |
| B R1 START_HEAD | `5e9878d71d363542d25ada19cff2590aa37448d3` |
| C R1 final | recorded in CTRL-X-R1 section of main qualification doc |

## Accounting rule

Every wide diagnostic is represented in `CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_GROUPS` with shared `(file, rule, semantic_owner, classification)`. `diagnostic_count` sums to `CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_TOTAL` (gate: `test_ctrl_x_wide_pyright_provenance_accounts_for_all_diagnostics`).

Non-boundary rows classify as `FRZ-TYP/EBH-6` with `future_owner=EBH-6` per CTRL-X-R1 non-blocking wide error rule.
