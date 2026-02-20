# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List

from intergrax.eval.eval_result import EvalResult


class EvalReportRenderer:
    """
    Deterministic renderer for evaluation results (P0).

    Responsibilities:
        - Transform list of EvalResult into deterministic textual report.
        - No scoring heuristics.
        - No semantic evaluation.
        - No LLM usage.

    Pure formatting layer.
    """

    def render(self, results: List[EvalResult]) -> str:

        total_cases: int = len(results)
        total_success: int = sum(1 for r in results if r.success)
        total_failure: int = total_cases - total_success
        total_tokens: int = sum(r.total_tokens for r in results)
        total_cost: float = sum(r.total_cost for r in results)

        lines: List[str] = []

        lines.append("EVAL REPORT")
        lines.append(f"Total cases: {total_cases}")
        lines.append(f"Success: {total_success}")
        lines.append(f"Failure: {total_failure}")
        lines.append(f"Total tokens: {total_tokens}")
        lines.append(f"Total cost: {total_cost:.2f}")
        lines.append("")

        for r in results:
            lines.append(f"Case: {r.case_id}")
            lines.append(f"  Success: {r.success}")
            lines.append(f"  Tokens: {r.total_tokens}")
            lines.append(f"  Cost: {r.total_cost:.2f}")
            lines.append(f"  Tool calls: {r.tool_calls_count}")
            if r.error is not None:
                lines.append(f"  Error: {r.error}")
            lines.append("")

        return "\n".join(lines) + "\n"