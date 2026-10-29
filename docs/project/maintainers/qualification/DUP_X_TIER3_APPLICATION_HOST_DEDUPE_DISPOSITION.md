# DUP-X — Tier-3 application host dedupe disposition (DUP-02)

| Field | Value |
| --- | --- |
| **Gap** | DUPX-GAP-L3-TIER3-DEDUPE |
| **Candidate** | DUPX-CAND-12 |
| **Wave** | DUP-X-03 Wave 1 |
| **Host discovery** | `applications/*/host/factory.py` (mechanical census @ Wave 1 HEAD) |

## Application hosts (9)

| Application | Host factory | Dedupe repository | Disposition |
| --- | --- | --- | --- |
| `attestation_demo` | yes | none | **N/A** — no product dedupe store |
| `dispute_sim_application` | yes | none | **N/A** |
| `governed_contractor_application` | yes | none | **N/A** |
| `intergrax_assistant_application` | yes | none | **N/A** |
| `lab_application` | yes | none | **N/A** |
| `legal_application` | yes | none | **N/A** |
| `local_workspace_application` | yes | `SlackEventDedupeRepository` | **DISTINCT** — LKW Slack Events API product dedupe |
| `poc_template_application` | yes | none | **N/A** |
| `research_application` | yes | none | **N/A** |

## LKW Slack dedupe (sole Tier-3 dedupe owner)

| Field | Value |
| --- | --- |
| **Module** | `applications/local_workspace_application/slack_companion/dedupe_repository.py` |
| **Wiring** | `wire_slack_companion` ← `host/factory.py` when daemon lifecycle enabled |
| **Store** | DocumentStore partition `lkw.slack_companion:dedupe` |
| **Truth** | Slack `team_id` + `event_id` product concurrency — not platform RF-STA-01 lifecycle |
| **Platform dedupe** | Context `exact_dedup` / memory contracts remain authoritative for harness semantics (CAND-06/07) |

## Conclusion

No shadow platform dedupe copy across Tier-3 hosts. Single bounded product repository in LKW Slack companion — **DISTINCT** responsibility, not duplicate platform owner.
