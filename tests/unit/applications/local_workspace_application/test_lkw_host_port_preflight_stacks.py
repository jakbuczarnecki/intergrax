# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PREFLIGHT_SCRIPT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/lkw_host_port_preflight.py"
)
_DOCKER_DIR = (
    _REPO_ROOT / "applications/local_workspace_application/docker"
)


def _load_preflight() -> ModuleType:
    module_name = "lkw_host_port_preflight_stacks_test"
    spec = importlib.util.spec_from_file_location(module_name, _PREFLIGHT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight() -> ModuleType:
    return _load_preflight()


def test_known_stack_definitions_include_product_and_proof_stacks(
    preflight: ModuleType,
) -> None:
    stacks = preflight.known_intergrax_stack_definitions(_DOCKER_DIR)
    stack_ids = {stack.stack_id for stack in stacks}
    assert stack_ids == {
        "lkw-product-quickstart",
        "lkw-core-platform-proof",
        "lkw-trusted-ask-workspace-proof",
    }
    product = next(stack for stack in stacks if stack.is_product_stack)
    assert product.compose_project == "intergrax_lkw"


def test_non_destructive_down_commands_for_known_stacks(
    preflight: ModuleType,
) -> None:
    stacks = preflight.known_intergrax_stack_definitions(_DOCKER_DIR)
    for stack in stacks:
        command = preflight.non_destructive_compose_down_args(stack)
        assert preflight.lifecycle_command_is_non_destructive(command)
        assert "down" in command
        assert "-v" not in command
        assert "--volumes" not in command


def test_classify_port_ownership_distinguishes_product_and_proof_stacks(
    preflight: ModuleType,
) -> None:
    owned = {
        "lkw-product-quickstart": frozenset({8020}),
        "lkw-core-platform-proof": frozenset({4318}),
        "lkw-trusted-ask-workspace-proof": frozenset(),
    }
    product = preflight.classify_port_ownership(
        8020,
        owned,
        product_stack_id="lkw-product-quickstart",
    )
    proof = preflight.classify_port_ownership(
        4318,
        owned,
        product_stack_id="lkw-product-quickstart",
    )
    assert product.kind == preflight.PortOwnershipKind.PRODUCT_STACK
    assert proof.kind == preflight.PortOwnershipKind.KNOWN_INTERGRAX_STACK
    assert proof.stack_id == "lkw-core-platform-proof"
