# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

from intergrax.llm.messages import ChatMessage

_SCHEMA_CONTEXT = """\
You are investigating parcel delivery delays using read-only SQL against one table.

Table: proof.parcel_events
Columns:
- parcel_id (bigint, primary key)
- created_at (timestamptz)
- region (text)
- origin_hub (text)
- destination_hub (text)
- carrier (text)
- service_type (text): standard, express, economy
- route_type (text): local, regional, long_haul
- distance_km (numeric)
- weight_kg (numeric)
- planned_hours (numeric)
- actual_hours (numeric)
- delayed (boolean)
- weekday (smallint)

Use the bounded read-only SQL tool to gather evidence before concluding.
Do not assume hidden causes; support claims with query results.

Before each tool call, include exactly two lines in your assistant message (auditable justification only):
EVIDENCE_BASIS: <comma-separated prior tool call_id values such as call_abc123 from earlier assistant tool_calls; leave empty on the first tool round only>
PURPOSE: <short purpose of the next SQL check>

On follow-up rounds after tool results exist, EVIDENCE_BASIS must cite the exact evidence_call_id values echoed in prior tool observations (for example call_abc123). Never use the SQL tool name in EVIDENCE_BASIS. Do not leave EVIDENCE_BASIS empty after the first tool round.
"""


def build_investigation_messages(*, question: str) -> list[ChatMessage]:
    """Model-safe context — schema and question only; no planted ground truth."""
    return [
        ChatMessage(role="system", content=_SCHEMA_CONTEXT),
        ChatMessage(role="user", content=question.strip()),
    ]


FORBIDDEN_PROMPT_SUBSTRINGS: tuple[str, ...] = (
    "ANOMALY_SEGMENT",
    "ANOMALY_PARCEL_MODULUS",
    "TRUE_ANOMALY_RATE",
    "verify_dataset_invariants",
    "ground_truth_version",
    "dataset.py",
    "bulk_load_parcel_events",
    "North-Volume hub dominates naive",
    "Planted structure",
)
