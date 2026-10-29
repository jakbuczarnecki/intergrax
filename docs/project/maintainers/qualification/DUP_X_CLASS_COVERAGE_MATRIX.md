# DUP-X — Class Coverage Matrix (DUP-01..14)

| Class | Discovery queries | Inspected paths (representative) | Result |
| --- | --- | --- | --- |
| **DUP-01** | `class \w+Registry` census; tools/context registries | `intergrax/tools/registry/`, `intergrax/context/`, `tools_base.py` | **finding** CAND-01 N/A; CAND-13 retained DISTINCT |
| **DUP-02** | Tier-3 platform copies | `applications/**` dedupe repositories | **PASS** CAND-12 N/A — LKW DISTINCT @ Wave 1 |
| **DUP-03** | Integration registry vs projection | `integrations/registry/*`, `runtime/integrations/registry_v2.py` | **finding** CAND-02 THIN adapter |
| **DUP-04** | dispatch / delegate census | `runtime/nexus/`, vendor knowledge sync | **N/A WITH EVIDENCE** CAND-05 pipeline stages |
| **DUP-05** | sqlite / dedup stores | EBH-6 sqlite annex + context dedup | **N/A WITH EVIDENCE** CAND-06/07/14 |
| **DUP-06** | retry / replay owners | `enterprise_reliability/`, recovery packages | **N/A WITH EVIDENCE** — single family owner |
| **DUP-07** | admit / policy middleware | `runtime/governance/` | **N/A WITH EVIDENCE** CAND-11 diagnostics bounded |
| **DUP-08** | bypass imports | lab paths, wiring shims | **finding** CAND-15 RATIFIED fence |
| **DUP-09** | deprecated / compat | `tools_base.py`, compat fixtures | **N/A WITH EVIDENCE** — zero prod imports |
| **DUP-10** | env / profile selection | `integrations/registry/factory.py`, profiles | **PASS** DUPX-GAP-CLASS-D @ Wave 1 caller matrix |
| **DUP-11** | marketplace qualified resolvers | `intergrax/tools/**` | **N/A WITH EVIDENCE** CAND-05 |
| **DUP-12** | diagnostics reconstructing truth | observability diagnostics registries | **RATIFIED** CAND-09 DISTINCT export routing |
| **DUP-13** | parallel DTOs / contract copies | `contracts/**`, `registry_v2` projection | **finding** CAND-02 delta dossier |
| **DUP-14** | composition hint duplication | COMPOSITION_ROOT_HINTS, TRACE-X, CONFIG-X | **PASS** qual SSOT unified (CAND-08) |

**Negative search record:** mechanical playbook in `_closed_world_inventory.class_coverage_playbook()` · integrity test `test_dup_x_closed_world_integrity`.
