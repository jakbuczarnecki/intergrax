# © Artur Czarnecki. All rights reserved.

"""PDC-D3-R004 — failover exhaustion is bounded (existing behaviour, proved adversarially)."""

from __future__ import annotations

import pytest
from intergrax.llm_adapters.contracts.failover_policy import FailoverDecision
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter, LLMRoutingAttemptRecord
from intergrax.llm_adapters.registry.failover_policy import default_failover_policy
from tests.qualification.pdc_x.d3.fakes import USER, AlwaysTryNextPolicy, BufferedOnlyAdapter, FailingAdapter

pytestmark = pytest.mark.unit


class _ServerError(Exception):
    status_code = 503


def _chain(n: int, *, error: BaseException | None = None) -> tuple[FailoverLLMAdapter, list[FailingAdapter]]:
    members = [FailingAdapter(error or _ServerError(f"down-{i}"), provider=f"p{i}") for i in range(n)]
    return FailoverLLMAdapter(members, failover_policy=AlwaysTryNextPolicy()), members


@pytest.mark.parametrize("size", [1, 2, 5, 12])
def test_exhaustion_tries_each_member_exactly_once_and_raises_the_last_error(size: int) -> None:
    chain, members = _chain(size)
    with pytest.raises(_ServerError) as caught:
        chain.generate_messages(USER)
    assert str(caught.value) == f"down-{size - 1}"
    assert [m.calls for m in members] == [["generate_messages"]] * size
    assert len(chain.routing_attempts) == size


def test_a_policy_that_always_asks_for_the_next_candidate_cannot_loop_past_the_chain() -> None:
    """Worst-case policy: bound comes from the chain, never from the policy."""
    chain, members = _chain(3)
    for _ in range(4):  # repeated calls must not accumulate attempts or calls beyond one pass each
        with pytest.raises(_ServerError):
            chain.generate_messages(USER)
        assert len(chain.routing_attempts) == 3
    assert [len(m.calls) for m in members] == [4, 4, 4]


def test_attempt_records_are_ordered_and_identify_each_failed_member() -> None:
    chain, _ = _chain(3)
    with pytest.raises(_ServerError):
        chain.generate_messages(USER)
    records = chain.routing_attempts
    assert all(isinstance(r, LLMRoutingAttemptRecord) for r in records)
    assert [r.profile_index for r in records] == [0, 1, 2]
    assert [r.provider for r in records] == ["p0", "p1", "p2"]
    assert all(r.error.startswith("_ServerError") for r in records)


def test_default_policy_stops_on_a_non_retriable_error() -> None:
    members = [FailingAdapter(ValueError("bad request"), provider=f"p{i}") for i in range(3)]
    chain = FailoverLLMAdapter(members, failover_policy=default_failover_policy())
    with pytest.raises(ValueError):
        chain.generate_messages(USER)
    assert [len(m.calls) for m in members] == [1, 0, 0]


def test_default_policy_fails_over_on_retriable_errors_and_is_bounded() -> None:
    members = [FailingAdapter(_ServerError(f"down-{i}"), provider=f"p{i}") for i in range(3)]
    chain = FailoverLLMAdapter(members, failover_policy=default_failover_policy())
    with pytest.raises(_ServerError) as caught:
        chain.generate_messages(USER)
    assert str(caught.value) == "down-2"
    assert [len(m.calls) for m in members] == [1, 1, 1]


def test_failover_recovers_on_a_later_member_without_touching_the_rest() -> None:
    first = FailingAdapter(_ServerError("down"), provider="p0")
    healthy = BufferedOnlyAdapter(provider="p1")
    never = BufferedOnlyAdapter(provider="p2")
    chain = FailoverLLMAdapter([first, healthy, never], failover_policy=AlwaysTryNextPolicy())
    assert chain.generate_messages(USER).content == "buffered"
    assert first.calls == ["generate_messages"]
    assert healthy.calls == ["generate_messages"]
    assert never.calls == []
    assert len(chain.routing_attempts) == 1


def test_stop_decision_ends_the_pass_immediately() -> None:
    class _Stop:
        def decide_after_failure(self, context: object) -> FailoverDecision:
            return FailoverDecision.STOP

    members = [FailingAdapter(_ServerError("x"), provider=f"p{i}") for i in range(3)]
    chain = FailoverLLMAdapter(members, failover_policy=_Stop())
    with pytest.raises(_ServerError):
        chain.generate_messages(USER)
    assert [len(m.calls) for m in members] == [1, 0, 0]


def test_empty_chain_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        FailoverLLMAdapter([], failover_policy=AlwaysTryNextPolicy())
