# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SHARED = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/run-lkw-os-interaction-proof.py"
)
_CORE = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/run-lkw-core-platform-proof.py"
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def interaction() -> ModuleType:
    return _load(_SHARED, "lkw_os_interaction_mongodb_stack")


@pytest.fixture(scope="module")
def core() -> ModuleType:
    return _load(_CORE, "lkw_core_mongodb_stack")


def test_managed_is_default_for_interaction(interaction: ModuleType) -> None:
    args = interaction._parse_args(["--os-family", "windows"])
    assert args.mongodb_stack == "managed"


def test_managed_is_default_for_core(core: ModuleType) -> None:
    parser = core.build_parser()
    args = parser.parse_args(
        ["--os-family", "windows", "--wrapper-id", "windows_bat"]
    )
    assert args.mongodb_stack == "managed"


def test_external_mode_never_invokes_docker(
    interaction: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def _forbidden(*_a: object, **_k: object) -> None:
        calls.append((_a, _k))
        raise AssertionError("docker_should_not_run")

    monkeypatch.setattr(interaction.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(interaction, "_run_command", _forbidden)
    monkeypatch.setattr(
        interaction,
        "verify_mongodb_reachable_via_platform",
        lambda: None,
    )
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", "mongodb://example.invalid:27017/db")
    interaction.prepare_mongodb(stack="external", mongo_express_url="http://x")
    assert calls == []


def test_external_mode_requires_uri(
    interaction: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    with pytest.raises(RuntimeError, match="external_mongodb_uri_required"):
        interaction.prepare_external_mongodb(mongo_express_url="http://x")


def test_external_mode_unreachable_fails_closed(
    interaction: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", "mongodb://example.invalid:27017/db")
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", "intergrax_proofs")
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", "proof_receipts")

    def _boom() -> tuple[object, object]:
        raise RuntimeError("external_mongodb_unreachable")

    monkeypatch.setattr(interaction, "resolve_mongodb_document_store", _boom)
    with pytest.raises(RuntimeError, match="external_mongodb_unreachable"):
        interaction.verify_mongodb_reachable_via_platform()


def test_external_mode_still_records_receipts(
    interaction: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[object] = []

    class _Store:
        def get(self, *_a: object, **_k: object) -> None:
            return None

        def close(self) -> None:
            return None

    class _Integration:
        pass

    def _resolve() -> tuple[object, object]:
        return _Integration(), _Store()

    def _record(receipt: object, *_a: object, **_k: object) -> object:
        recorded.append(receipt)
        return receipt

    monkeypatch.setattr(interaction, "resolve_mongodb_document_store", _resolve)
    monkeypatch.setattr(interaction, "record_and_verify_proof_receipt", _record)
    receipt = object()
    verified, _integration = interaction.record_os_interaction_proof_receipt(receipt)
    assert verified is receipt
    assert recorded == [receipt]


def test_no_receipt_bypass_flags_in_parsers(
    interaction: ModuleType,
    core: ModuleType,
) -> None:
    interaction_help = interaction._parse_args.__doc__ or ""
    core_help = core.build_parser().format_help()
    for text in (interaction_help, core_help):
        assert "--skip-receipt" not in text
        assert "--no-receipt" not in text
        assert "--trust-external" not in text
        assert "--ignore-verification" not in text
    source_interaction = _SHARED.read_text(encoding="utf-8")
    source_core = _CORE.read_text(encoding="utf-8")
    for source in (source_interaction, source_core):
        assert "--skip-receipt" not in source
        assert "--no-receipt" not in source
        assert "--trust-external" not in source
        assert "--ignore-verification" not in source


def test_core_external_application_hosting_skips_docker_requirement(
    core: ModuleType,
) -> None:
    config = core.ProofConfig(
        os_family=core.OsFamily.LINUX,
        wrapper_id=core.WrapperId.LINUX_SH,
        phase="application-hosting",
        run_id_prefix="lkw-core-",
        base_url="http://127.0.0.1:8020",
        kafka_ui="http://127.0.0.1:8085",
        mongo_express="http://127.0.0.1:8086",
        elasticsearch_url="http://127.0.0.1:9200",
        kibana_url="http://127.0.0.1:5601",
        sentry_url="http://127.0.0.1:9000",
        phase_timeout_seconds=30,
        mongodb_stack="external",
    )
    assert core.phase_requires_docker(config) is False


def test_core_managed_still_requires_docker(core: ModuleType) -> None:
    config = core.ProofConfig(
        os_family=core.OsFamily.WINDOWS,
        wrapper_id=core.WrapperId.WINDOWS_BAT,
        phase="application-hosting",
        run_id_prefix="lkw-core-",
        base_url="http://127.0.0.1:8020",
        kafka_ui="http://127.0.0.1:8085",
        mongo_express="http://127.0.0.1:8086",
        elasticsearch_url="http://127.0.0.1:9200",
        kibana_url="http://127.0.0.1:5601",
        sentry_url="http://127.0.0.1:9000",
        phase_timeout_seconds=30,
        mongodb_stack="managed",
    )
    assert core.phase_requires_docker(config) is True
