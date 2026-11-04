# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from scripts.governance.verify_commit_ticket_refs import (
    CommitRefConfig,
    message_has_valid_ref,
    path_touches_product,
)


@pytest.fixture
def config() -> CommitRefConfig:
    return CommitRefConfig(
        product_path_prefixes=("intergrax/", "agents/", "applications/", "scripts/"),
        ref_line_prefix="Ref: governance/tickets/",
        ref_id_pattern=r"T-IGX-[0-9]{3}-[A-Z]\.md",
        skip_env="INTERGRAX_SKIP_TICKET_REF_GATE",
    )


def test_path_touches_product(config: CommitRefConfig) -> None:
    assert path_touches_product("intergrax/foo.py", config.product_path_prefixes)
    assert path_touches_product("docs/readme.md", config.product_path_prefixes) is False


def test_valid_ref_trailer(config: CommitRefConfig) -> None:
    msg = "feat: thing\n\nRef: governance/tickets/T-IGX-001-A.md\n"
    assert message_has_valid_ref(msg, config)


def test_invalid_ref_missing(config: CommitRefConfig) -> None:
    assert message_has_valid_ref("feat: no ref\n", config) is False


def test_invalid_ref_wrong_path(config: CommitRefConfig) -> None:
    msg = "Ref: governance/tickets/wrong.md\n"
    assert message_has_valid_ref(msg, config) is False


def test_invalid_ref_duplicate(config: CommitRefConfig) -> None:
    msg = "Ref: governance/tickets/T-IGX-001-A.md\nRef: governance/tickets/T-IGX-002-A.md\n"
    assert message_has_valid_ref(msg, config) is False
