# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CraftResultPromoter — typed promotion L0 validation (ECC-3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.codecraft.contracts import CodeCraftSession, CraftResult, StaticGateResult
from intergrax.runtime.events.payload_registry import get_payload_schema

_SCHEMA_VALIDATION_ERROR = "promotion_schema_validation_failed"

SchemaResolver = Callable[[str], type[BaseModel] | None]


class CraftPromotionPayload(BaseModel):
    """Default promotion schema when profile has no custom ref."""

    model_config = ConfigDict(extra="forbid")

    craft_id: str
    goal: str
    stdout: str = ""
    code: str = ""
    success: bool = False


@dataclass(frozen=True, slots=True)
class CraftPromotionEligibility:
    """Evidence-backed promotion eligibility — never caller-asserted."""

    eligible: bool
    error: str = ""


class CraftResultPromoter:
    """Validate and export craft output for pipeline handoff."""

    def __init__(self, *, schema_resolver: SchemaResolver | None = None) -> None:
        self._schema_resolver: SchemaResolver = schema_resolver or get_payload_schema

    def assess_promotion_eligibility(self, session: CodeCraftSession) -> CraftPromotionEligibility:
        if session.disposed:
            return CraftPromotionEligibility(eligible=False, error="craft_session_disposed")
        if session.promoted:
            return CraftPromotionEligibility(eligible=False, error="craft_already_promoted")
        if not session.iterations:
            return CraftPromotionEligibility(eligible=False, error="promotion_verification_missing")
        last = session.iterations[-1]
        if not last.static_gate.passed:
            return CraftPromotionEligibility(eligible=False, error="static_gate_failed")
        if not last.exec_success:
            return CraftPromotionEligibility(eligible=False, error="execution_not_verified")
        if last.test_passed is False:
            return CraftPromotionEligibility(eligible=False, error="tests_not_passed")
        if last.verdict != "promote":
            return CraftPromotionEligibility(eligible=False, error="cvl_verdict_not_promote")
        if not bool(session.structured_output.get("success", False)):
            return CraftPromotionEligibility(eligible=False, error="promotion_verification_missing")
        return CraftPromotionEligibility(eligible=True)

    def promote_session(
        self,
        session: CodeCraftSession,
        *,
        schema_ref: str | None = None,
    ) -> CraftResult:
        eligibility = self.assess_promotion_eligibility(session)
        if not eligibility.eligible:
            gate = StaticGateResult(
                passed=False,
                rule_ids=[eligibility.error or "promotion_denied"],
                message=eligibility.error or "promotion_denied",
            )
            return CraftResult(
                craft_id=session.craft_id,
                success=False,
                mode=session.mode,
                static_gate=gate,
                error=eligibility.error or "promotion_denied",
                verdict="abort",
            )

        payload = CraftPromotionPayload(
            craft_id=session.craft_id,
            goal=session.goal,
            stdout=str(session.structured_output.get("stdout") or ""),
            code=session.code,
            success=True,
        )
        try:
            structured = self._validate_payload(payload, schema_ref=schema_ref)
        except ValueError:
            return self._deny_schema_validation(session)
        return CraftResult(
            craft_id=session.craft_id,
            success=True,
            mode=session.mode,
            static_gate=StaticGateResult(passed=True, rule_ids=[]),
            stdout=str(structured.get("stdout") or ""),
            structured_output=structured,
            sandbox_session_id=session.sandbox_session_id,
            verdict="promote",
        )

    def _validate_payload(
        self,
        payload: CraftPromotionPayload,
        *,
        schema_ref: str | None,
    ) -> dict[str, object]:
        if schema_ref is None:
            return payload.model_dump()
        if not schema_ref.strip():
            raise ValueError(_SCHEMA_VALIDATION_ERROR)
        schema_cls = self._schema_resolver(schema_ref)
        if schema_cls is None:
            raise ValueError(_SCHEMA_VALIDATION_ERROR)
        try:
            validated = schema_cls.model_validate(payload.model_dump())
        except ValidationError as exc:
            raise ValueError(f"{_SCHEMA_VALIDATION_ERROR}: {exc}") from exc
        return validated.model_dump()

    @staticmethod
    def _deny_schema_validation(session: CodeCraftSession) -> CraftResult:
        return CraftResult(
            craft_id=session.craft_id,
            success=False,
            mode=session.mode,
            static_gate=StaticGateResult(
                passed=False,
                rule_ids=[_SCHEMA_VALIDATION_ERROR],
                message=_SCHEMA_VALIDATION_ERROR,
            ),
            error=_SCHEMA_VALIDATION_ERROR,
            verdict="abort",
        )
