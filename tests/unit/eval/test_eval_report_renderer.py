# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from intergrax.eval.eval_report_renderer import EvalReportRenderer
from intergrax.eval.eval_result import EvalResult


def test_render_deterministic_report():

    renderer = EvalReportRenderer()

    results = [
        EvalResult(
            case_id="case-1",
            success=True,
            final_answer="a",
            total_tokens=10,
            total_cost=10.0,
            tool_calls_count=0,
            error=None,
        ),
        EvalResult(
            case_id="case-2",
            success=False,
            final_answer="b",
            total_tokens=20,
            total_cost=20.0,
            tool_calls_count=1,
            error="Mismatch",
        ),
    ]

    report = renderer.render(results)

    expected = (
        "EVAL REPORT\n"
        "Total cases: 2\n"
        "Success: 1\n"
        "Failure: 1\n"
        "Total tokens: 30\n"
        "Total cost: 30.00\n"
        "\n"
        "Case: case-1\n"
        "  Success: True\n"
        "  Tokens: 10\n"
        "  Cost: 10.00\n"
        "  Tool calls: 0\n"
        "\n"
        "Case: case-2\n"
        "  Success: False\n"
        "  Tokens: 20\n"
        "  Cost: 20.00\n"
        "  Tool calls: 1\n"
        "  Error: Mismatch\n"
        "\n"
    )

    assert report == expected