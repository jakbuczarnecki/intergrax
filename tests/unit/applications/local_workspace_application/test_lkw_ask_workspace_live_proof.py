# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/run-lkw-ask-workspace-live-proof.py"
)


def _load_module() -> ModuleType:
    module_name = "run_lkw_ask_workspace_live_proof"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def proof() -> ModuleType:
    return _load_module()


def test_compose_command_uses_explicit_proof_project(proof: ModuleType) -> None:
    args = proof._compose_command("ps")
    assert args[:4] == ["docker", "compose", "-p", proof._COMPOSE_PROJECT]
    assert proof._COMPOSE_PROJECT == "lkw-trusted-ask-workspace-proof"
    assert str(proof._BASE_COMPOSE) in args
    assert str(proof._MONGODB_COMPOSE) in args
    assert str(proof._TRUSTED_ASK_PROOF_COMPOSE) in args


def test_compose_lifecycle_commands_include_trusted_ask_proof_overlay(
    proof: ModuleType,
) -> None:
    overlay = str(proof._TRUSTED_ASK_PROOF_COMPOSE)
    for command in (
        proof._compose_command("config"),
        proof._compose_command("up", "-d"),
        proof._compose_command("exec", "-T", "local_workspace", "printenv"),
        proof._compose_command("restart", *proof._RESTART_SERVICES),
        proof._compose_command("ps", "-a"),
    ):
        assert overlay in command


def test_trusted_ask_proof_overlay_mounts_sample_docs_read_only(
    proof: ModuleType,
) -> None:
    overlay = yaml.safe_load(proof._TRUSTED_ASK_PROOF_COMPOSE.read_text(encoding="utf-8"))
    volumes = overlay["services"]["local_workspace"]["volumes"]
    assert "../sample_docs:/data/user_docs:ro" in volumes


def test_trusted_ask_proof_overlay_allowlist_is_container_path_only(
    proof: ModuleType,
) -> None:
    overlay = yaml.safe_load(proof._TRUSTED_ASK_PROOF_COMPOSE.read_text(encoding="utf-8"))
    allowlist = overlay["services"]["local_workspace"]["environment"][
        "INTERGRAX_ALLOWED_READ_ROOTS"
    ]
    assert allowlist == "/data/user_docs"
    assert "\\" not in allowlist
    assert re.search(r"^[A-Za-z]:", allowlist) is None


def test_host_proof_document_path_matches_mounted_sample_docs_dir(
    proof: ModuleType,
) -> None:
    assert proof._SAMPLE_DOCS_DIR == proof._APP_DIR / "sample_docs"
    assert proof._SAMPLE_DOCS_DIR.resolve() == (proof._DOCKER_DIR / "../sample_docs").resolve()
    proof_file_name = "ask_qdrant_durability_test.txt"
    host_doc_path = proof._SAMPLE_DOCS_DIR / proof_file_name
    container_doc_path = f"/data/user_docs/{proof_file_name}"
    assert host_doc_path.name == Path(container_doc_path).name
    assert str(host_doc_path.parent.resolve()) == str(proof._SAMPLE_DOCS_DIR.resolve())


def test_compose_paths_include_project_for_lifecycle_commands(proof: ModuleType) -> None:
    for command in (
        proof._compose_command("up", "-d"),
        proof._compose_command("config"),
        proof._compose_command("exec", "-T", "local_workspace", "printenv"),
        proof._compose_command("restart", *proof._RESTART_SERVICES),
        proof._compose_command("ps", "-a"),
    ):
        assert command[3] == proof._COMPOSE_PROJECT


def test_restart_targets_only_proof_project(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_compose(*args: str, **kwargs: Any) -> object:
        captured.append(proof._compose_command(*args))
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(proof, "_run_compose", fake_run_compose)
    proof.restart_lkw_and_qdrant()
    assert len(captured) == 1
    command = captured[0]
    assert command[3] == proof._COMPOSE_PROJECT
    assert "restart" in command
    assert "local_workspace" in command
    assert "qdrant" in command


def test_occupied_foreign_port_fails_before_compose_up(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail_preflight() -> None:
        events.append("preflight")
        raise proof.ProofFailure("port_preflight", "required_port_unavailable:8020")

    monkeypatch.setattr(proof, "check_startup_host_port_preflight", fail_preflight)
    monkeypatch.setattr(
        proof,
        "start_canonical_stack",
        lambda: events.append("compose_up"),
    )

    with pytest.raises(proof.ProofFailure):
        proof.check_startup_host_port_preflight()
    assert events == ["preflight"]
    assert "compose_up" not in events


def test_product_quickstart_port_collision_has_safe_reason(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proof,
        "resolve_compose_published_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(
        proof,
        "canonical_compose_owned_host_ports",
        lambda *, compose_exec_args, **_k: (
            frozenset({8020})
            if proof._PRODUCT_COMPOSE_PROJECT in compose_exec_args("ps")[3]
            else frozenset()
        ),
    )
    monkeypatch.setattr(proof, "is_loopback_tcp_port_reachable", lambda _port: True)
    monkeypatch.setattr(proof, "probe_host_port_available", lambda _port: False)

    with pytest.raises(proof.ProofFailure) as exc:
        proof.check_startup_host_port_preflight()
    assert exc.value.phase == "port_preflight"
    assert "lkw_product_quickstart" in exc.value.reason


def test_core_platform_port_collision_has_safe_reason(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proof,
        "resolve_compose_published_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(
        proof,
        "canonical_compose_owned_host_ports",
        lambda *, compose_exec_args, **_k: (
            frozenset({8020})
            if proof._CORE_PLATFORM_COMPOSE_PROJECT in compose_exec_args("ps")[3]
            else frozenset()
        ),
    )
    monkeypatch.setattr(proof, "is_loopback_tcp_port_reachable", lambda _port: True)
    monkeypatch.setattr(proof, "probe_host_port_available", lambda _port: False)

    with pytest.raises(proof.ProofFailure) as exc:
        proof.check_startup_host_port_preflight()
    assert exc.value.phase == "port_preflight"
    assert "lkw_core_platform_proof" in exc.value.reason


def test_proof_owned_ports_are_not_rejected(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proof,
        "resolve_compose_published_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(
        proof,
        "canonical_compose_owned_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(proof, "is_loopback_tcp_port_reachable", lambda _port: True)
    monkeypatch.setattr(proof, "probe_host_port_available", lambda _port: False)

    proof.check_startup_host_port_preflight()


def test_verify_running_vector_store_uses_proof_compose_project(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_compose(*args: str, **kwargs: Any) -> object:
        captured.append(proof._compose_command(*args))
        stdout = "qdrant\n" if "printenv" in args else "LOCAL_WORKSPACE_VECTOR_STORE: qdrant\n"
        return type("CP", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(proof, "_run_compose", fake_run_compose)
    provider = proof.verify_running_vector_store_is_qdrant()
    assert provider == "qdrant"
    assert captured
    for command in captured:
        assert command[3] == proof._COMPOSE_PROJECT


def _fake_typed_profile_compose(
    *,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str,
    embedding_model: str,
    pulled: list[str],
) -> object:
    def fake_run_compose(*args: str, **kwargs: Any) -> object:
        if args[:3] == ("exec", "-T", "local_workspace") and "python" in args:
            code = args[-1]
            if "llm_profile_from_env" in code:
                stdout = f"{llm_provider}\n{llm_model}\n"
            elif "embedding_profile_from_env" in code:
                stdout = f"{embedding_provider}\n{embedding_model}\n"
            else:
                stdout = ""
            return type("CP", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()
        if "pull" in args:
            pulled.append(args[-1])
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    return fake_run_compose


def test_trusted_ask_ensure_ollama_model_pulls_chat_and_embedding_models(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []
    monkeypatch.setattr(
        proof,
        "_run_compose",
        _fake_typed_profile_compose(
            llm_provider="ollama",
            llm_model="llama3.1:latest",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            pulled=pulled,
        ),
    )
    proof.ensure_ollama_model()
    assert pulled == ["llama3.1:latest", "nomic-embed-text"]


def test_trusted_ask_skips_embedding_pull_when_provider_not_ollama(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []
    monkeypatch.setattr(
        proof,
        "_run_compose",
        _fake_typed_profile_compose(
            llm_provider="ollama",
            llm_model="llama3.1:latest",
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            pulled=pulled,
        ),
    )
    proof.ensure_ollama_model()
    assert pulled == ["llama3.1:latest"]


def test_trusted_ask_skips_generation_pull_when_provider_not_ollama(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []
    monkeypatch.setattr(
        proof,
        "_run_compose",
        _fake_typed_profile_compose(
            llm_provider="openai",
            llm_model="gpt-4o",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            pulled=pulled,
        ),
    )
    proof.ensure_ollama_model()
    assert pulled == ["nomic-embed-text"]


def test_trusted_ask_generation_and_embedding_providers_are_independent(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []
    monkeypatch.setattr(
        proof,
        "_run_compose",
        _fake_typed_profile_compose(
            llm_provider="openai",
            llm_model="gpt-4o",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            pulled=pulled,
        ),
    )
    proof.ensure_ollama_model()
    assert "gpt-4o" not in pulled
    assert pulled == ["nomic-embed-text"]


def test_trusted_ask_skips_all_ollama_pulls_when_neither_provider_is_ollama(
    proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []
    monkeypatch.setattr(
        proof,
        "_run_compose",
        _fake_typed_profile_compose(
            llm_provider="openai",
            llm_model="gpt-4o",
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            pulled=pulled,
        ),
    )
    proof.ensure_ollama_model()
    assert pulled == []


def test_env_example_utf8_without_mojibake_keeps_canonical_embedding_pair() -> None:
    path = _REPO_ROOT / "applications/local_workspace_application/.env.example"
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert "â" not in text
    assert "INTERGRAX_EMBEDDING_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text" in text
