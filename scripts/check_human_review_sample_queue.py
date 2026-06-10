#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-25.2 — human review sample queue gate."""

from __future__ import annotations

import sys

from intergrax.runtime.evaluation.human_review_sample_queue import HumanReviewSampleQueue


def main() -> int:
    queue = HumanReviewSampleQueue()
    sample = queue.enqueue(
        run_id="run_test",
        agent_id="echo",
        scenario_id="golden-echo",
        reason="shadow_eval_borderline",
    )
    pending = queue.list_pending()
    if len(pending) != 1:
        print("queue must retain pending sample", file=sys.stderr)
        return 1
    reviewed = queue.mark_reviewed(sample.sample_id, reviewer_id="reviewer@ops")
    if reviewed is None or not reviewed.reviewed:
        print("mark_reviewed failed", file=sys.stderr)
        return 1
    if queue.list_pending():
        print("reviewed sample must not remain pending", file=sys.stderr)
        return 1

    print("OK: human review sample queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
