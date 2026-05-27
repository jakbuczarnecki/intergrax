# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CLI formatters for experiment registry."""

from __future__ import annotations

from typing import List

from intergrax.experiments.models import ExperimentRecord


def format_experiment_list(records: List[ExperimentRecord]) -> str:
    if not records:
        return "No experiments found."
    lines = [
        f"{'EXPERIMENT_ID':<34} {'DECISION':<10} {'CAPABILITY':<24} HYPOTHESIS",
        "-" * 100,
    ]
    for record in records:
        hypothesis = record.hypothesis
        if len(hypothesis) > 40:
            hypothesis = hypothesis[:37] + "..."
        lines.append(
            f"{record.experiment_id:<34} {record.decision.value:<10} "
            f"{record.capability:<24} {hypothesis}"
        )
    return "\n".join(lines)


def format_experiment_show(record: ExperimentRecord) -> str:
    lines = [
        f"experiment_id:       {record.experiment_id}",
        f"capability:          {record.capability}",
        f"agent_id:            {record.agent_id or '(none)'}",
        f"decision:            {record.decision.value}",
        f"created_at:          {record.created_at_utc}",
        f"updated_at:          {record.updated_at_utc}",
        f"hypothesis:          {record.hypothesis}",
        f"expected_output:     {record.expected_output or '(none)'}",
        f"validation_criteria: {record.validation_criteria or '(none)'}",
        f"notes:               {record.notes or '(none)'}",
        f"run_ids:             {', '.join(record.run_ids) if record.run_ids else '(none)'}",
    ]
    return "\n".join(lines)
