# T-IGX-065-A — PT-T01: Campaign B tracked negative tests NT-01..NT-16

## Approved

Approved: operator @ 2026-11-23T13:30:00+01:00

## Scope

PT-T01 on development: registry SSOT tests/unit/campaign_b/_pt_t01_matrix.py. Held-out NT-04b, NT-07, NT-15e.

## Plan digest

Plan digest: 2d916c97680a7d1c86ba836fbdcea3cfe42bde11c4df838108fb2d5f39023279

## Lifecycle state

in_progress — P-R0-6: PT-T01 matrix active; **not closed**

## PT-T01 reconciliation (@ development)

- Registry tests/unit/campaign_b: 23 PASS (incl. NT-02 contract note).
- Active matrix: 88 PASS.
- NT-02: tracked GREEN; alpha pp_gov v2 FAIL expected (bare agent vs host-injected posture).
- HELD_OUT: NT-04b, NT-07 -> T-IGX-051-J; NT-15e -> PT-WF follow-on.
- NT-08: PT-WD owns test_wd_changed_since_default_deny.py.