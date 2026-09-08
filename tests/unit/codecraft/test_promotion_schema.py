# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7B-GATE-01 — promotion_schema_ref resolution and validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ConfigDict

from intergrax.codecraft.contracts import CodeCraftSession, IterationRecord, StaticGateResult
from intergrax.codecraft.promoter import CraftPromotionPayload, CraftResultPromoter
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads import RuntimeEventPayload

pytestmark = pytest.mark.unit

TENANT = "tenant-a"
TASK = "task-a"
KNOWN_SCHEMA_REF = "agents.codecraft_test.promotion.valid.v1"
STRICT_SCHEMA_REF = "agents.codecraft_test.promotion.strict.v1"


class _ValidPromotionPayloadV1(RuntimeEventPayload):
    schema_id = KNOWN_SCHEMA_REF

    craft_id: str
    goal: str
    stdout: str = ""
    code: str = ""
    success: bool = False


class _StrictPromotionPayloadV1(RuntimeEventPayload):
    schema_id = STRICT_SCHEMA_REF

    model_config = ConfigDict(extra="forbid", frozen=True)

    craft_id: str
    goal: str
    stdout: str = ""
    code: str = ""
    success: bool = False
    required_tag: str


@pytest.fixture(autouse=True)
def _register_promotion_schemas() -> None:
    register_payload_schema(_ValidPromotionPayloadV1, extension=True)
    register_payload_schema(_StrictPromotionPayloadV1, extension=True)


def _eligible_session(**updates: object) -> CodeCraftSession:
    base = CodeCraftSession(
        craft_id="craft-promo",
        task_id=TASK,
        tenant_id=TENANT,
        goal="demo",
        mode="autonomous",
        code="print('x')\n",
        structured_output={"success": True, "stdout": "x\n"},
        iterations=[
            IterationRecord(
                iteration=1,
                static_gate=StaticGateResult(passed=True),
                exec_success=True,
                test_passed=True,
                verdict="promote",
            ),
        ],
    )
    return base.model_copy(update=updates)


def test_promotion_schema_ref_none_uses_default_payload() -> None:
    promoter = CraftResultPromoter()
    result = promoter.promote_session(_eligible_session(), schema_ref=None)
    assert result.success is True
    assert result.verdict == "promote"
    assert result.static_gate.passed is True
    assert result.structured_output == CraftPromotionPayload(
        craft_id="craft-promo",
        goal="demo",
        stdout="x\n",
        code="print('x')\n",
        success=True,
    ).model_dump()


def test_promotion_schema_ref_blank_denied() -> None:
    promoter = CraftResultPromoter()
    for blank in ("", "   "):
        result = promoter.promote_session(_eligible_session(), schema_ref=blank)
        assert result.success is False
        assert result.verdict != "promote"
        assert result.static_gate.passed is False
        assert result.error == "promotion_schema_validation_failed"


def test_promotion_schema_ref_unknown_denied() -> None:
    promoter = CraftResultPromoter()
    result = promoter.promote_session(_eligible_session(), schema_ref="unknown-schema-ref")
    assert result.success is False
    assert result.verdict != "promote"
    assert result.static_gate.passed is False
    assert result.error == "promotion_schema_validation_failed"


def test_promotion_schema_ref_known_invalid_payload_denied() -> None:
    promoter = CraftResultPromoter()
    result = promoter.promote_session(_eligible_session(), schema_ref=STRICT_SCHEMA_REF)
    assert result.success is False
    assert result.verdict != "promote"
    assert result.static_gate.passed is False
    assert result.error == "promotion_schema_validation_failed"


def test_promotion_schema_ref_known_valid_payload_promoted() -> None:
    promoter = CraftResultPromoter()
    result = promoter.promote_session(_eligible_session(), schema_ref=KNOWN_SCHEMA_REF)
    assert result.success is True
    assert result.verdict == "promote"
    assert result.static_gate.passed is True
    assert result.structured_output["craft_id"] == "craft-promo"
    assert result.structured_output["success"] is True


def test_promotion_schema_resolver_invoked_once_with_exact_ref() -> None:
    resolver = MagicMock(return_value=_ValidPromotionPayloadV1)
    promoter = CraftResultPromoter(schema_resolver=resolver)
    result = promoter.promote_session(_eligible_session(), schema_ref=KNOWN_SCHEMA_REF)
    resolver.assert_called_once_with(KNOWN_SCHEMA_REF)
    assert result.success is True
    assert result.verdict == "promote"
