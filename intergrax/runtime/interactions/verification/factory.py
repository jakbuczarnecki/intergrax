# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for inbound request verifiers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from intergrax.runtime.interactions.verification.contract import (
    InboundRequestVerifier,
    NullInboundRequestVerifier,
)
from intergrax.runtime.interactions.verification.slack_signature import (
    ENV_SLACK_VERIFY_SIGNATURE,
    resolve_slack_signing_secret,
)
from intergrax.runtime.interactions.verification.teams_signature import (
    ENV_TEAMS_VERIFY_SIGNATURE,
    resolve_teams_security_token,
)

ENV_INBOUND_VERIFIER = "INTERGRAX_INBOUND_VERIFIER"


class InboundVerifierMode(str, Enum):
    NONE = "none"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass(frozen=True)
class InboundVerifierSettings:
    mode: InboundVerifierMode = InboundVerifierMode.NONE
    slack_signing_secret: str = ""
    slack_verify_enabled: Optional[bool] = None
    teams_security_token: str = ""
    teams_verify_enabled: Optional[bool] = None


def resolve_inbound_verifier_settings(
    *,
    mode: Optional[str] = None,
    slack_signing_secret: Optional[str] = None,
    slack_verify_enabled: Optional[bool] = None,
    teams_security_token: Optional[str] = None,
    teams_verify_enabled: Optional[bool] = None,
) -> InboundVerifierSettings:
    raw_mode = (mode or os.environ.get(ENV_INBOUND_VERIFIER, InboundVerifierMode.NONE.value)).strip().lower()
    try:
        resolved_mode = InboundVerifierMode(raw_mode)
    except ValueError:
        resolved_mode = InboundVerifierMode.NONE
    return InboundVerifierSettings(
        mode=resolved_mode,
        slack_signing_secret=resolve_slack_signing_secret(slack_signing_secret),
        slack_verify_enabled=slack_verify_enabled,
        teams_security_token=resolve_teams_security_token(teams_security_token),
        teams_verify_enabled=teams_verify_enabled,
    )


def create_inbound_verifier(
    settings: Optional[InboundVerifierSettings] = None,
    *,
    implementation: Optional[InboundRequestVerifier] = None,
) -> InboundRequestVerifier:
    if implementation is not None:
        return implementation
    resolved = settings or resolve_inbound_verifier_settings()
    if resolved.mode == InboundVerifierMode.SLACK:
        from intergrax.integrations.providers.slack.bundle import create_slack_signature_verifier

        return create_slack_signature_verifier(
            signing_secret=resolved.slack_signing_secret,
            enabled=resolved.slack_verify_enabled,
        )
    if resolved.mode == InboundVerifierMode.TEAMS:
        from intergrax.integrations.providers.teams.bundle import create_teams_signature_verifier

        return create_teams_signature_verifier(
            security_token=resolved.teams_security_token,
            enabled=resolved.teams_verify_enabled,
        )
    return NullInboundRequestVerifier()
