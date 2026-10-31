# © Artur Czarnecki. All rights reserved.

"""EBH-7 Batch C — RPL-H semantic evidence integrity gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._rpl_h_semantic_evidence import (
    RplSemanticVerdict,
    all_rpl_concern_semantic_evidence,
    rpl_semantic_verdict_counts,
)
from tests.qualification.ebh_7._rpl_h_semantic_review import assert_ebh7_rpl_h_semantic_review_complete

pytestmark = [pytest.mark.qualification]


def test_ebh7_batch_c_rpl_semantic_evidence_covers_matrix() -> None:
    assert_ebh7_rpl_h_semantic_review_complete()
    evidence = all_rpl_concern_semantic_evidence()
    assert len(evidence) == 56


def test_ebh7_batch_c_rpl_semantic_verdict_counts_recorded() -> None:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_verdict_counts

    agent = rpl_semantic_verdict_counts()
    norm = normative_rpl_verdict_counts()
    assert sum(agent.values()) == 56
    assert agent == norm
    assert agent[RplSemanticVerdict.BLOCKED] == 6
    assert agent[RplSemanticVerdict.PARTIAL] >= 17
