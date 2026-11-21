# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for inbound request verifiers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from intergrax.core.security.security_configuration import (
    INBOUND_VERIFIER_DISABLED,
    INBOUND_VERIFIER_MODE_INVALID,
    INBOUND_VERIFIER_MODE_UNSET,
    INBOUND_VERIFIER_NONE_FORBIDDEN,
    INBOUND_VERIFIER_SLACK_SECRET_MISSING,
    INBOUND_VERIFIER_SLACK_VERIFY_DISABLED,
    INBOUND_VERIFIER_TEAMS_TOKEN_MISSING,
    SecurityConfigurationError,
)
from intergrax.integrations.contracts.base import LabSelectorOptIn
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


class InboundVerifierMode(StrEnum):
    NONE = "none"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass(frozen=True)
class InboundVerifierSettings:
    mode: InboundVerifierMode = InboundVerifierMode.NONE
    slack_signing_secret: str = ""
    slack_verify_enabled: bool | None = None
    teams_security_token: str = ""
    teams_verify_enabled: bool | None = None


def resolve_inbound_verifier_settings(
    *,
    mode: str | None = None,
    slack_signing_secret: str | None = None,
    slack_verify_enabled: bool | None = None,
    teams_security_token: str | None = None,
    teams_verify_enabled: bool | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> InboundVerifierSettings:
    if mode is not None:
        raw_mode = mode.strip().lower()
    else:
        env_mode = os.environ.get(ENV_INBOUND_VERIFIER)
        if env_mode is not None and str(env_mode).strip():
            raw_mode = str(env_mode).strip().lower()
        elif lab_opt_in is not None:
            raw_mode = lab_opt_in.selector.strip().lower()
        else:
            raise SecurityConfigurationError(
                item=ENV_INBOUND_VERIFIER,
                reason_code=INBOUND_VERIFIER_MODE_UNSET,
                message="inbound verifier mode is unset",
            )
    try:
        resolved_mode = InboundVerifierMode(raw_mode)
    except ValueError as exc:
        raise SecurityConfigurationError(
            item=ENV_INBOUND_VERIFIER,
            reason_code=INBOUND_VERIFIER_MODE_INVALID,
            message=f"unknown inbound verifier mode {raw_mode!r}",
        ) from exc
    return InboundVerifierSettings(
        mode=resolved_mode,
        slack_signing_secret=resolve_slack_signing_secret(slack_signing_secret),
        slack_verify_enabled=slack_verify_enabled,
        teams_security_token=resolve_teams_security_token(teams_security_token),
        teams_verify_enabled=teams_verify_enabled,
    )


def _require_verifier_operational(
    verifier: InboundRequestVerifier,
    *,
    mode: InboundVerifierMode,
    settings: InboundVerifierSettings,
    lab_opt_in: LabSelectorOptIn | None,
) -> None:
    enabled = getattr(verifier, "enabled", True)
    if enabled:
        return
    if lab_opt_in is not None:
        return
    if mode == InboundVerifierMode.SLACK:
        if not settings.slack_signing_secret.strip():
            raise SecurityConfigurationError(
                item=ENV_SLACK_VERIFY_SIGNATURE,
                reason_code=INBOUND_VERIFIER_SLACK_SECRET_MISSING,
                message="Slack signing secret is required for inbound verification",
            )
        raise SecurityConfigurationError(
            item=ENV_SLACK_VERIFY_SIGNATURE,
            reason_code=INBOUND_VERIFIER_SLACK_VERIFY_DISABLED,
            message="Slack inbound signature verification is disabled",
        )
    if mode == InboundVerifierMode.TEAMS:
        from intergrax.runtime.interactions.verification.teams_signature import ENV_TEAMS_SECURITY_TOKEN

        if not settings.teams_security_token.strip():
            raise SecurityConfigurationError(
                item=ENV_TEAMS_SECURITY_TOKEN,
                reason_code=INBOUND_VERIFIER_TEAMS_TOKEN_MISSING,
                message="Teams security token is required for inbound verification",
            )
        raise SecurityConfigurationError(
            item=ENV_TEAMS_VERIFY_SIGNATURE,
            reason_code=INBOUND_VERIFIER_DISABLED,
            message="Teams inbound signature verification is disabled",
        )


def create_inbound_verifier(
    settings: InboundVerifierSettings | None = None,
    *,
    implementation: InboundRequestVerifier | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> InboundRequestVerifier:
    if implementation is not None:
        return implementation
    resolved = settings or resolve_inbound_verifier_settings(lab_opt_in=lab_opt_in)
    if resolved.mode == InboundVerifierMode.NONE:
        if lab_opt_in is None:
            raise SecurityConfigurationError(
                item=ENV_INBOUND_VERIFIER,
                reason_code=INBOUND_VERIFIER_NONE_FORBIDDEN,
                message="inbound verifier mode 'none' requires explicit laboratory opt-in",
            )
        return NullInboundRequestVerifier()
    if resolved.mode == InboundVerifierMode.SLACK:
        from intergrax.integrations.providers.notification_channel.slack.bundle import create_slack_signature_verifier

        verifier = create_slack_signature_verifier(
            signing_secret=resolved.slack_signing_secret,
            enabled=resolved.slack_verify_enabled,
        )
        _require_verifier_operational(
            verifier,
            mode=resolved.mode,
            settings=resolved,
            lab_opt_in=lab_opt_in,
        )
        return verifier
    if resolved.mode == InboundVerifierMode.TEAMS:
        from intergrax.integrations.providers.notification_channel.teams.bundle import create_teams_signature_verifier

        verifier = create_teams_signature_verifier(
            security_token=resolved.teams_security_token,
            enabled=resolved.teams_verify_enabled,
        )
        _require_verifier_operational(
            verifier,
            mode=resolved.mode,
            settings=resolved,
            lab_opt_in=lab_opt_in,
        )
        return verifier
    if lab_opt_in is None:
        raise SecurityConfigurationError(
            item=ENV_INBOUND_VERIFIER,
            reason_code=INBOUND_VERIFIER_NONE_FORBIDDEN,
            message="inbound verifier mode requires explicit laboratory opt-in",
        )
    return NullInboundRequestVerifier()
