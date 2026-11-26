# PLAN T-IGX-077-C — resume classification is not a directory prefix

A directory prefix is not `A_CANONICAL_RESUME`. A file outside the inventory is `G_UNCLEAR`. Inventoried files keep `A_CANONICAL_RESUME`.

Write set:

- `tests/qualification/trace_x/_trace_x_p6_restart_resume_classification.py`
- one test module under `tests/qualification/trace_x/` that states the uninventoried path is `G_UNCLEAR` and inventoried files stay `A_CANONICAL_RESUME`

No product edit. No roadmap edit. No FRZ checklist edit.

Tests: the new test plus TRACE gates q02, q08, and q11 in one pytest command.
