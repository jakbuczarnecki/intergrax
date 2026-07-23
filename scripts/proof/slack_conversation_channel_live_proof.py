#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Controlled live proof for Slack conversation-channel Socket Mode runtime.

Requires:
  INTERGRAX_SLACK_APP_TOKEN
  INTERGRAX_SLACK_BOT_TOKEN

Loads optional credentials from applications/local_workspace_application/.env
(process environment still wins). Does not print tokens, message text, or full
Slack payloads.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from intergrax.integrations.contracts.conversation_channel import (
    ConversationChoiceOption,
    ConversationEventKind,
    ConversationSingleChoice,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    ENV_APP_TOKEN,
    ENV_BOT_TOKEN,
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LKW_ENV_FILE = _REPO_ROOT / "applications" / "local_workspace_application" / ".env"
_ACTION_ID = "intergrax_runtime_proof"


def _load_lkw_environment(env_file: Path = _LKW_ENV_FILE) -> bool:
    """Load LKW application ``.env`` when present. Returns whether the file exists."""
    if not env_file.is_file():
        return False
    load_dotenv(env_file, override=False)
    return True


_LKW_ENV_AVAILABLE = _load_lkw_environment()
_PROOF_TIMEOUT_SECONDS = float(os.environ.get("INTERGRAX_SLACK_PROOF_TIMEOUT_SECONDS", "180"))


def _configuration_source_message(*, env_available: bool = _LKW_ENV_AVAILABLE) -> str:
    if env_available:
        return "[proof] configuration source: LKW .env available"
    return "[proof] configuration source: process environment / no LKW .env"


class _ProofState:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.message_done = asyncio.Event()
        self.action_done = asyncio.Event()
        self.integration: SlackConversationChannelIntegration | None = None

    def mark(self, label: str) -> None:
        self.events.append(label)
        print(f"[proof] {label}", flush=True)


async def _handler(state: _ProofState, event: InboundConversationEvent) -> None:
    assert state.integration is not None
    if event.kind is ConversationEventKind.MESSAGE:
        state.mark("MESSAGE delivered to handler")
        await state.integration.send(
            OutboundConversationMessage(
                address=event.address,
                text="Intergrax Slack conversation runtime received your message.",
            )
        )
        state.mark("thread reply created")
        await state.integration.send(
            OutboundConversationMessage(
                address=event.address,
                text="Select one proof option:",
                components=(
                    ConversationSingleChoice(
                        action_id=_ACTION_ID,
                        prompt="Choose",
                        options=(
                            ConversationChoiceOption(value="option_a", label="Option A"),
                            ConversationChoiceOption(value="option_b", label="Option B"),
                        ),
                    ),
                ),
            )
        )
        state.mark("single-choice message created")
        state.message_done.set()
        return

    if event.kind is ConversationEventKind.ACTION:
        state.mark("ACTION delivered to handler")
        selected = event.action.selected_value if event.action is not None else ""
        label = "Option A" if selected == "option_a" else "Option B" if selected == "option_b" else "unknown"
        await state.integration.send(
            OutboundConversationMessage(
                address=event.address,
                text=f"Selected: {label}",
            )
        )
        state.mark("action confirmation sent")
        state.action_done.set()


async def _run() -> int:
    print(_configuration_source_message(), flush=True)
    if not os.environ.get(ENV_APP_TOKEN) or not os.environ.get(ENV_BOT_TOKEN):
        print("LIVE_PROOF_BLOCKED: missing INTERGRAX_SLACK_APP_TOKEN and/or INTERGRAX_SLACK_BOT_TOKEN")
        return 2

    config = SlackConversationChannelIntegrationConfig.from_env(enabled=True)
    integration = SlackConversationChannelIntegration.from_config(config)
    state = _ProofState()
    state.integration = integration

    async def handler(event: InboundConversationEvent) -> None:
        await _handler(state, event)

    state.mark("starting Socket Mode")
    await integration.start(handler)
    health = integration.health()
    if not health.healthy:
        print(f"LIVE_PROOF_BLOCKED: health not ready ({health.detail})")
        await integration.stop()
        return 3
    state.mark("connection established")
    print(
        f"[proof] ready — send one DM, then select an option (timeout={_PROOF_TIMEOUT_SECONDS:.0f}s)",
        flush=True,
    )

    deadline = time.monotonic() + _PROOF_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if state.message_done.is_set() and state.action_done.is_set():
                break
            await asyncio.sleep(0.25)
        else:
            print("LIVE_PROOF_BLOCKED: timed out waiting for MESSAGE+ACTION sequence")
            return 4
    finally:
        await integration.stop()
        state.mark("clean stop completed")

    required = {
        "connection established",
        "MESSAGE delivered to handler",
        "thread reply created",
        "single-choice message created",
        "ACTION delivered to handler",
        "clean stop completed",
    }
    missing = sorted(required - set(state.events))
    if missing:
        print(f"LIVE_PROOF_BLOCKED: missing evidence: {missing}")
        return 5

    print("[proof] SUMMARY=PASS")
    for item in state.events:
        print(f"[proof] evidence: {item}")
    return 0


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("LIVE_PROOF_BLOCKED: interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 — proof harness boundary
        print(f"LIVE_PROOF_BLOCKED: {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
