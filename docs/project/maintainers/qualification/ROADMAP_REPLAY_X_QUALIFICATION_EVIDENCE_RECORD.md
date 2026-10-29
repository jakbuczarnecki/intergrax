# ROADMAP-REPLAY-X — Qualification evidence record (@ IMPLEMENTATION_SHA)

| Field | Value |
| --- | --- |
| **START_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **IMPLEMENTATION_SHA** | `c6504da3ebcadb631bae7b6d18eb54153acc1fbe` |
| **TEST_EXECUTION_SHA** | `c6504da3ebcadb631bae7b6d18eb54153acc1fbe` |
| **EVIDENCE_PUBLICATION_SHA** | **PENDING** — L2/L3 not complete |
| **Environment** | Windows dev · `uv` · repo root `D:\Projekty\intergrax` |

## L1 — mechanical (@ `c6504da3`)

| Command | Exit | Result | Log |
| --- | ---: | --- | --- |
| `uv run pytest tests/qualification/roadmap_replay_x/ -q` | 0 | **15 passed** | `.tmp/proof/roadmap-replay-x-pub-c6504da3/l1-mechanical.log` |

## L2 — parent replay

| Batch | Command scope | Exit | Result | Log |
| --- | --- | ---: | --- | --- |
| DUP-X | `tests/qualification/dup_x/` | 0 | **44 passed** | `l2-dup-x.log` |
| Parents core | config+compat+trace+state+ctrl+gov+ebh_5 | 2 | **4 collection ERRORs** (STATE-X closed-world inventory) | `l2-parents-core.log` |

**Root cause (collection):** `state_x._state_x_closed_world_durable_state_support` discovers durable classes under  
`applications/governed_contractor_application/docker/runtime-context/intergrax/...` → **unclassified durable mechanisms** (local GCA docker qualification tree; not introduced by REPLAY-X diff).

**Impact:** STATE-X / TRACE-X L2 batches **BLOCKED** on this machine until inventory scope excludes docker runtime-context or mechanisms classified.

**Remaining L2 (OPEN):** tenant_x, ebh_6 selective, NPSC allowlist expected-FAIL, recovered config/compat/ctrl/gov/ebh_5 batch.

## L3 — ECAP bundle

**OPEN** — requires green L2 per manifest.

## Status

**QUALIFICATION INCOMPLETE** — do not issue Protocol 03 B6 PASS. Implementation sources **published** @ `c6504da3`.
