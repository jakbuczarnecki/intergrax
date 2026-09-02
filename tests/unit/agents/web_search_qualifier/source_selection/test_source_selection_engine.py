# © Artur Czarnecki. All rights reserved.

"""Unit tests for SourceSelectionEngine policy ordering and LLM fallback."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from web_search_qualifier.source_selection.contracts import (
    SourceSelectionContext,
    SourceSelectionMode,
    SourceSelectionOutcome,
    SourceSelectionPolicyDecision,
    SourceSelectionPolicyDescriptor,
    SourceSelectionPolicyId,
)
from web_search_qualifier.source_selection.engine import (
    SourceSelectionContractError,
    SourceSelectionEngine,
)
from web_search_qualifier.source_selection.example_docs_policy import ExampleDocsSourceSelectionPolicy
from web_search_qualifier.source_selection.llm_selector import LLMSourceSelector
from web_search_qualifier.web_search import WebSearchCandidate

pytestmark = pytest.mark.unit

_DOCS_URL = "https://docs.example.com/guide"
_OTHER_URL = "https://vendor.example.net/page"
_PYTHON_FINAL = "https://www.python.org/downloads/release/python-3120/"
_PYTHON_RC = "https://www.python.org/downloads/release/python-3120rc3"


class _StubPolicy:
    def __init__(
        self,
        *,
        policy_id: str,
        decision: SourceSelectionPolicyDecision,
    ) -> None:
        self._descriptor = SourceSelectionPolicyDescriptor(
            policy_id=SourceSelectionPolicyId(policy_id),
            display_name=policy_id,
        )
        self._decision = decision

    @property
    def descriptor(self) -> SourceSelectionPolicyDescriptor:
        return self._descriptor

    def evaluate(self, context: SourceSelectionContext) -> SourceSelectionPolicyDecision:
        del context
        return self._decision


class _StubLLMAdapter:
    def __init__(self, *, content: str) -> None:
        self._content = content
        self.called = False

    def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
        del messages, temperature, run_id
        self.called = True
        return type("R", (), {"content": self._content})()


def _candidate(url: str, *, rank: int = 1) -> WebSearchCandidate:
    return WebSearchCandidate(
        rank=rank,
        url=url,
        title=url,
        snippet="snippet",
        provider="test",
    )


def test_first_policy_select_skips_llm() -> None:
    adapter = _StubLLMAdapter(content=_OTHER_URL)
    engine = SourceSelectionEngine(
        policies=(
            _StubPolicy(
                policy_id="source.first",
                decision=SourceSelectionPolicyDecision(
                    outcome=SourceSelectionOutcome.SELECT,
                    selected_url=_DOCS_URL,
                    reason_code="first",
                ),
            ),
        ),
        llm_selector=LLMSourceSelector(adapter=adapter, system_prompt="select"),
    )
    decision = engine.select(
        run_id="run-1",
        task_message="docs guide",
        candidates=(_candidate(_DOCS_URL), _candidate(_OTHER_URL, rank=2)),
    )
    assert decision.selected_url == _DOCS_URL
    assert decision.provenance.selection_mode is SourceSelectionMode.POLICY
    assert adapter.called is False


def test_second_policy_selects_when_first_abstains() -> None:
    adapter = _StubLLMAdapter(content=_OTHER_URL)
    engine = SourceSelectionEngine(
        policies=(
            _StubPolicy(
                policy_id="source.first",
                decision=SourceSelectionPolicyDecision(
                    outcome=SourceSelectionOutcome.ABSTAIN,
                    reason_code="abstain",
                ),
            ),
            _StubPolicy(
                policy_id="source.second",
                decision=SourceSelectionPolicyDecision(
                    outcome=SourceSelectionOutcome.SELECT,
                    selected_url=_OTHER_URL,
                    reason_code="second",
                ),
            ),
        ),
        llm_selector=LLMSourceSelector(adapter=adapter, system_prompt="select"),
    )
    decision = engine.select(
        run_id="run-1",
        task_message="task",
        candidates=(_candidate(_OTHER_URL),),
    )
    assert decision.selected_url == _OTHER_URL
    assert decision.provenance.policy_id is not None
    assert decision.provenance.policy_id.value == "source.second"
    assert adapter.called is False


def test_all_abstain_invokes_llm() -> None:
    adapter = _StubLLMAdapter(content=_OTHER_URL)
    engine = SourceSelectionEngine(
        policies=(
            _StubPolicy(
                policy_id="source.first",
                decision=SourceSelectionPolicyDecision(
                    outcome=SourceSelectionOutcome.ABSTAIN,
                    reason_code="abstain",
                ),
            ),
        ),
        llm_selector=LLMSourceSelector(adapter=adapter, system_prompt="select"),
    )
    decision = engine.select(
        run_id="run-1",
        task_message="task",
        candidates=(_candidate(_OTHER_URL),),
    )
    assert decision.selected_url == _OTHER_URL
    assert decision.provenance.selection_mode is SourceSelectionMode.LLM
    assert adapter.called is True


def test_policy_selecting_unavailable_candidate_raises() -> None:
    engine = SourceSelectionEngine(
        policies=(
            _StubPolicy(
                policy_id="source.bad",
                decision=SourceSelectionPolicyDecision(
                    outcome=SourceSelectionOutcome.SELECT,
                    selected_url="https://missing.example/",
                    reason_code="bad",
                ),
            ),
        ),
        llm_selector=None,
    )
    with pytest.raises(SourceSelectionContractError):
        engine.select(
            run_id="run-1",
            task_message="task",
            candidates=(_candidate(_OTHER_URL),),
        )


def test_policy_order_is_deterministic() -> None:
    first = _StubPolicy(
        policy_id="source.first",
        decision=SourceSelectionPolicyDecision(
            outcome=SourceSelectionOutcome.SELECT,
            selected_url=_DOCS_URL,
            reason_code="first",
        ),
    )
    second = _StubPolicy(
        policy_id="source.second",
        decision=SourceSelectionPolicyDecision(
            outcome=SourceSelectionOutcome.SELECT,
            selected_url=_OTHER_URL,
            reason_code="second",
        ),
    )
    engine = SourceSelectionEngine(policies=(first, second), llm_selector=None)
    decision = engine.select(
        run_id="run-1",
        task_message="task",
        candidates=(_candidate(_DOCS_URL), _candidate(_OTHER_URL, rank=2)),
    )
    assert decision.provenance.policy_id is not None
    assert decision.provenance.policy_id.value == "source.first"


def test_example_docs_policy_works_without_engine_change() -> None:
    engine = SourceSelectionEngine(
        policies=(ExampleDocsSourceSelectionPolicy(),),
        llm_selector=None,
    )
    decision = engine.select(
        run_id="run-1",
        task_message="read the guide",
        candidates=(
            _candidate(_OTHER_URL, rank=1),
            _candidate(_DOCS_URL, rank=2),
        ),
    )
    assert decision.selected_url == _DOCS_URL
    assert decision.provenance.policy_id is not None
    assert decision.provenance.policy_id.value == "source.example_docs_preference"
