# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy, trim_message_to_budget


@pytest.mark.unit
def test_trim_message_to_budget_truncates() -> None:
    policy = ContextBudgetPolicy(max_chars=10)
    result = trim_message_to_budget("hello world!!!", policy)
    assert result.trimmed is True
    assert len(result.message) == 10
    assert result.original_chars > result.final_chars
