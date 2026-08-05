from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.proofs.config import (
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofArtifactError,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    UniversalTokenOptimizationProofRunner,
)
from scripts.token_optimization.run_universal_proof import (
    EXIT_INVALID_CONFIG,
    EXIT_OK,
    EXIT_PROVIDER_UNAVAILABLE,
    main,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 5, 7, 0, tzinfo=UTC)


def _config(tmp_path: Path, *, mode: str = "offline_smoke") -> Path:
    path = tmp_path / "proof.toml"
    path.write_text(
        f"""
schema_version = "token-optimization-proof.v1"
proof_id = "artifact-proof"
run_mode = "{mode}"

[adapter]
adapter_id = "adapter"
provider = "vllm"
type = "openai_compatible"
model = "safe-model"
base_url = "http://127.0.0.1:8100/v1"
api_key_env = "ARTIFACT_PROOF_KEY"
timeout_seconds = 5.0
max_output_tokens = 32
temperature = 0.0

[router]
enabled = true
configuration_id = "exact_only"
minimum_confidence = 0.6
allow_structured_output_fallback = true
require_review_for_protected_lossy_content = true

[pipeline]
mode = "replace"
layer_ids = ["builtin.exact_deduplication"]
failure_policy = "continue"

[output]
directory = ".artifacts/proof"
fail_if_exists = true

[[cases]]
case_id = "safe-case"
source_type = "prompt"
content = "TOP_SECRET_PROMPT\\nTOP_SECRET_PROMPT"
tags = ["smoke"]

[cases.policy]
enabled = true
profile = "balanced"
allow_lossy = false
require_validation = true
fallback_on_validation_failure = true
""",
        encoding="utf-8",
    )
    return path


def test_artifacts_are_canonical_redaction_safe_and_manifested(tmp_path: Path) -> None:
    config = load_universal_token_optimization_proof_config(_config(tmp_path))
    runner = UniversalTokenOptimizationProofRunner(
        clock=_fixed_clock,
        run_id_factory=lambda: "fixed-run",
    )
    output = tmp_path / "output"
    result = runner.run(config, output_directory=output, run_id="fixed-run")
    run_dir = output / "artifact-proof" / "fixed-run"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["raw_content_included"] is False
    assert run_payload["raw_content_included"] is False
    assert "TOP_SECRET_PROMPT" not in (run_dir / "run.json").read_text(encoding="utf-8")
    assert result.artifact_manifest.files
    for file_ref in manifest["files"]:
        path = run_dir / file_ref["path"]
        assert path.is_file()
        assert file_ref["sha256"]
    assert (run_dir / "cases" / "safe-case.json").is_file()
    assert (run_dir / "run.json").read_bytes().endswith(b"\n")


def test_checked_in_sample_uses_repository_root_artifact_directory(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "test-only")
    config = load_universal_token_optimization_proof_config(
        _REPO_ROOT / "configs" / "token_optimization" / "proof_vllm.toml"
    )

    assert config.output.directory == (
        _REPO_ROOT / ".artifacts" / "token_optimization" / "proof"
    )


def test_vllm_compose_is_loopback_bound_without_privileged_access() -> None:
    compose = (
        _REPO_ROOT / "infra" / "docker" / "vllm" / "docker-compose.yml"
    ).read_text(encoding="utf-8")
    readme = (_REPO_ROOT / "infra" / "docker" / "vllm" / "README.md").read_text(
        encoding="utf-8"
    )

    assert '127.0.0.1:8100:8000' in compose
    assert "8100:8000" in compose
    assert "127.0.0.1:8100:8000" in readme
    assert "privileged:" not in compose
    assert "docker.sock" not in compose


def test_duplicate_run_directory_fails_closed(tmp_path: Path) -> None:
    config = load_universal_token_optimization_proof_config(_config(tmp_path))
    runner = UniversalTokenOptimizationProofRunner()
    output = tmp_path / "output"
    runner.run(config, output_directory=output, run_id="same-run")

    with pytest.raises(ProofArtifactError, match="RUN_DIRECTORY_EXISTS"):
        runner.run(config, output_directory=output, run_id="same-run")


def test_fixed_clock_and_run_id_produce_stable_json(tmp_path: Path) -> None:
    config = load_universal_token_optimization_proof_config(_config(tmp_path))
    runner = UniversalTokenOptimizationProofRunner(clock=_fixed_clock)
    first = tmp_path / "first"
    second = tmp_path / "second"
    runner.run(config, output_directory=first, run_id="stable")
    runner.run(config, output_directory=second, run_id="stable")

    first_files = sorted(path.relative_to(first) for path in first.rglob("*.json"))
    second_files = sorted(path.relative_to(second) for path in second.rglob("*.json"))
    assert first_files == second_files
    for relative_path in first_files:
        assert (first / relative_path).read_bytes() == (
            second / relative_path
        ).read_bytes()


def test_cli_validate_offline_and_safe_provider_errors(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    offline_config = _config(tmp_path)
    assert (
        main(
            [
                "--config",
                str(offline_config),
                "--mode",
                "offline_smoke",
                "--validate-only",
            ]
        )
        == EXIT_OK
    )
    assert "valid=true" in capsys.readouterr().out

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("schema_version = [", encoding="utf-8")
    assert main(["--config", str(invalid)]) == EXIT_INVALID_CONFIG
    assert "INVALID_TOML" in capsys.readouterr().err

    monkeypatch.delenv("ARTIFACT_PROOF_KEY", raising=False)
    live_config = _config(tmp_path, mode="live_adapter")
    assert main(["--config", str(live_config), "--validate-only"]) == EXIT_PROVIDER_UNAVAILABLE
    captured = capsys.readouterr()
    assert "MISSING_API_KEY_ENV" in captured.err
    assert "TOP_SECRET_PROMPT" not in captured.out + captured.err
