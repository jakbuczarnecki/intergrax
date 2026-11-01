# EVOL-X W1 — Closed-world & coverage gate strengthening (Investigation B)

## Three buckets (must not conflate)

1. **Complete known critical dependencies** — edges in graph with MANDATORY extractors; oracle second method agrees.
2. **Explicit conservative unknowns** — `DEPENDS_UNKNOWN` / `unknown:path`; forces `CONSERVATIVE_FULL` selection; counts against M-02 denominator.
3. **Unresolved critical omissions** — oracle finds consumer/critical edge **not** in graph and **not** classified UNKNOWN with disposition → **CW FAIL**, blocks W2 exit.
4. **Out-of-scope with accepted evidence** — NA register + predicate X-*; never used to hide inbound edges from classified roots (CW-04).

## Selection vs PASS

| Condition | Selection behavior | FRZ / stage PASS |
|-----------|-------------------|------------------|
| INTEGRITY_FAILURE | CONSERVATIVE_FULL | **Cannot** claim FRZ-EVOL-06 PASS on that run |
| `critical_omissions > 0` | REFUSED_INTEGRITY | **Fail** closed-world gate |
| UNKNOWN only, no omission | CONSERVATIVE_FULL | May run tests; **M-02/M-02C** still fail until resolved |
| Empty selected set | Forbidden (R8) | Fail |

Conservative selection **does not** convert unresolved critical omission into qualified PASS.

## CW-01..10 W1 status

Algorithm **specified** in master plan v2 §5; mechanical engine **W2-05**. W1 delivers baseline numbers (`W1_DISCOVERY_BASELINE.md`), exclusion skeleton, parity targets vs COMPAT-X roots.
