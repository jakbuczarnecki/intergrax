# © Artur Czarnecki. All rights reserved.

"""MP-6E scoped Collaborative Activity read orchestration."""

from __future__ import annotations

from intergrax.collaborative_work.collaborative_activity_page_cursor_codec import (
    decode_collaborative_activity_page_cursor,
)
from intergrax.collaborative_work.collaborative_activity_read_authorization import (
    CollaborativeActivityReadAuthorizationEvaluator,
)
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityPage,
    CollaborativeActivityQuery,
    CollaborativeActivityReadPort,
)
from intergrax.contracts.collaborative_activity_read import (
    CollaborativeActivityCursorInvalid,
    CollaborativeActivityReadAuthorizationOutcome,
    CollaborativeActivityReadDenied,
    CollaborativeActivityReadDenialReason,
    CollaborativeActivityReadPolicyError,
    CollaborativeActivityReadRequest,
)


class CollaborativeActivityReadService:
    """Authorize every read independently, validate cursor, then query injected read port."""

    def __init__(
        self,
        *,
        read_authorization: CollaborativeActivityReadAuthorizationEvaluator,
        read_port: CollaborativeActivityReadPort,
    ) -> None:
        self._read_authorization = read_authorization
        self._read_port = read_port

    def read_page(self, request: CollaborativeActivityReadRequest) -> CollaborativeActivityPage:
        try:
            decision = self._read_authorization.evaluate(request)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise CollaborativeActivityReadPolicyError(
                "collaborative activity read authorization evaluation failed",
            ) from exc

        if decision.outcome is not CollaborativeActivityReadAuthorizationOutcome.ALLOW:
            reason = (
                decision.denial_reason
                or CollaborativeActivityReadDenialReason.POLICY_AMBIGUITY
            )
            raise CollaborativeActivityReadDenied(
                denial_reason=reason,
                policy_id=decision.policy_id,
            )

        authorized = decision.authorized_query
        assert authorized is not None
        self._validate_cursor_before_provider(authorized)
        return self._read_port.query(authorized)

    @staticmethod
    def _validate_cursor_before_provider(query: CollaborativeActivityQuery) -> None:
        if query.cursor is None:
            return
        try:
            decode_collaborative_activity_page_cursor(query.cursor, query=query)
        except CollaborativeActivityCursorInvalid:
            raise
