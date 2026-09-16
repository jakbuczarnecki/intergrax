# © Artur Czarnecki. All rights reserved.

"""External Work meaningful side-effect action identities (GEC-5 / GR-6-ARCH).

Values are ``DecisionExecutionActionKind``-compatible: the same string is used
in ``MeaningfulSideEffectRequest.action``, ``DecisionGovernanceMaterialRef.bound_action_kind``,
and Decision execution authorization — no runtime alias map.
"""

from __future__ import annotations

from typing import Final

from intergrax.contracts.decision_authorization import validate_decision_execution_action_kind


def _external_work_action_kind(value: str) -> str:
    return str(validate_decision_execution_action_kind(value))


ACTION_CREATE_EXTERNAL_WORK: Final = _external_work_action_kind("external_work.create")
ACTION_ACCEPT_QUOTE: Final = _external_work_action_kind("external_work.accept_quote")
ACTION_CANCEL_EXTERNAL_WORK: Final = _external_work_action_kind("external_work.cancel")
