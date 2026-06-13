# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CraftResultPromoter — typed promotion L0 validation (ECC-3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intergrax.codecraft.contracts import CodeCraftSession, CraftResult, StaticGateResult


class CraftPromotionPayload(BaseModel):
    """Default promotion schema when profile has no custom ref."""

    model_config = ConfigDict(extra="forbid")

    craft_id: str
    goal: str
    stdout: str = ""
    code: str = ""
    success: bool = False


class CraftResultPromoter:
    """Validate and export craft output for pipeline handoff."""

    def promote_session(
        self,
        session: CodeCraftSession,
        *,
        schema_ref: str | None = None,
    ) -> CraftResult:
        payload = CraftPromotionPayload(
            craft_id=session.craft_id,
            goal=session.goal,
            stdout=str(session.structured_output.get("stdout") or ""),
            code=session.code,
            success=bool(session.structured_output.get("success", False)),
        )
        structured = self._validate_payload(payload, schema_ref=schema_ref)
        session = session.model_copy(
            update={
                "promoted": True,
                "structured_output": structured,
                "status": "closed",
            },
        )
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

    @staticmethod
    def _validate_payload(payload: CraftPromotionPayload, *, schema_ref: str | None) -> dict[str, Any]:
        if schema_ref is None:
            return payload.model_dump()
        # Extension point: resolve schema_ref to Pydantic model via catalog.
        try:
            return payload.model_dump()
        except ValidationError as exc:
            raise ValueError(f"promotion_schema_validation_failed: {exc}") from exc
