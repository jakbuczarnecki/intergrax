# TOOL-LIFE-X R4 — Wave execution & audit bundle (partial)

| Field | Value |
| --- | --- |
| **START_HEAD** | `be09f5a89027f701bbcd8ace0e7185ec73f34b0e` |
| **F3 PUBLICATION_SHA** | `88d72cd7205e308b3601bd65aa5049d4f873861c` |
| **Disposition** | `TOOL-LIFE-X R4 — PARTIAL / EXACT BLOCKERS` |

## F3

| Item | Result |
| --- | --- |
| Prepublication EQG + L2 | **PASS** — 36 tests, log `pytest-f3-prepublication-l2.log` @ `88d72cd72` |
| Published | `origin/development` @ `88d72cd72` |
| Physical CANCEL-C HTTP L3 | **OPEN** |
| Production HTTP catalog | **Option C** (disabled) |

## F5 Gate A (G-A01–G-A09)

| ID | Result |
| --- | --- |
| G-A01–G-A06, G-A08 (scoped) | **PARTIAL** — reused L3 subset **75 passed** (`pytest-f5-l3-subset.log`); not full R4 per-sink race matrix @ new fencing |
| G-A07 | **BLOCKED** — proposal [`TOOL_LIFE_X_R4_G_A07_PRODUCTION_WIRING_PROPOSAL.md`](TOOL_LIFE_X_R4_G_A07_PRODUCTION_WIRING_PROPOSAL.md); no prod change |
| G-A09 positive | **BLOCKED** — F6 not run (OD-TL8-IMPL conditional + G-A07) |
| HTTP/MCP physical B-class fence | **Authorization-only** — honesty per R5-B |

## OD-EC7

[`TOOL_LIFE_X_R4_OD_EC7_RECORD.md`](TOOL_LIFE_X_R4_OD_EC7_RECORD.md) — **PASS scoped**; Batch C in L3 subset.

## F6 scenario 24

**NOT RUN** — Gate A parent closure incomplete (G-A07 + dedicated R4 revoke→deny L3 rows).

## EXT-L4

Independent audit required; no parent CLOSED.

## Logs (local)

`.tmp/session/TOOL-LIFE-X-R4/`
