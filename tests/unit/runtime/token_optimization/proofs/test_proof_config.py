from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.token_optimization.proofs.config import (
    _OPENAI_COMPATIBLE_PROVIDERS,
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofConfigurationError,
)


def _toml(
    *,
    proof_id: str = "proof-smoke",
    output: str = ".artifacts/proof",
    provider: str = "vllm",
    adapter_type: str = "openai_compatible",
    api_key_env: str | None = "SYNTHETIC_REQUIRED_API_KEY",
) -> str:
    api_key_line = (
        f'api_key_env = "{api_key_env}"\n' if api_key_env is not None else ""
    )
    return f"""
schema_version = "token-optimization-proof.v1"
proof_id = "{proof_id}"
run_mode = "offline_smoke"

[adapter]
adapter_id = "offline"
provider = "{provider}"
type = "{adapter_type}"
model = "offline-model"
base_url = "http://127.0.0.1:8100/v1"
{api_key_line}\
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
directory = "{output}"
fail_if_exists = true

[[cases]]
case_id = "case-one"
source_type = "prompt"
content = "secret prompt value\\nsecret prompt value"
tags = ["smoke"]

[cases.policy]
enabled = true
profile = "balanced"
allow_lossy = false
require_validation = true
fallback_on_validation_failure = true
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "proof.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_toml_loads_immutable_config(tmp_path: Path) -> None:
    config = load_universal_token_optimization_proof_config(
        _write(tmp_path, _toml())
    )

    assert config.proof_id == "proof-smoke"
    assert config.cases[0].case_id == "case-one"
    assert config.output.directory == (Path.cwd() / ".artifacts/proof").resolve()
    assert "secret prompt value" not in repr(config.cases[0])


def test_canonical_vllm_config_is_no_auth(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    config = load_universal_token_optimization_proof_config(
        Path(__file__).resolve().parents[5]
        / "configs"
        / "token_optimization"
        / "proof_vllm.toml"
    )

    assert config.adapter.api_key_env is None
    assert config.pipeline.layer_ids == ()


@pytest.mark.parametrize("provider", ("vllm", "openai", "groq"))
def test_openai_compatible_v1_profile_matches_canonical_registry(
    tmp_path: Path,
    provider: str,
) -> None:
    config = load_universal_token_optimization_proof_config(
        _write(tmp_path, _toml(provider=provider))
    )

    assert LLMProvider(provider).value == provider
    assert provider in LLMAdapterRegistry.registered_providers()
    assert provider in _OPENAI_COMPATIBLE_PROVIDERS
    assert config.adapter.adapter_type == "openai_compatible"


@pytest.mark.parametrize("provider", ("ollama", "claude", "unknown"))
def test_non_compatible_or_unknown_provider_is_rejected(
    tmp_path: Path,
    provider: str,
) -> None:
    with pytest.raises(ProofConfigurationError, match="UNSUPPORTED_ADAPTER"):
        load_universal_token_optimization_proof_config(
            _write(tmp_path, _toml(provider=provider))
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ('\nunknown = "reject"\n', "UNKNOWN_ROOT_FIELD"),
        ("proof_id = \" bad-id \"", "INVALID_PROOF_ID"),
        ("directory = 42", "INVALID_OUTPUT_DIRECTORY"),
        ("\nunknown = true\n", "UNKNOWN_CASE_POLICY_FIELD"),
    ],
)
def test_strict_fields_and_identifiers_fail_closed(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    text = _toml()
    if mutation.startswith("proof_id"):
        text = text.replace('proof_id = "proof-smoke"', mutation)
    elif mutation.startswith("directory"):
        text = text.replace('directory = ".artifacts/proof"', mutation)
    elif mutation == "\nunknown = true\n":
        text = text.replace(
            "fallback_on_validation_failure = true",
            "fallback_on_validation_failure = true" + mutation,
        )
    elif mutation.startswith("\nunknown ="):
        text = text.replace("\n[adapter]", mutation + "\n[adapter]")
    else:
        text = text.replace("\n[adapter]", mutation + "\n[adapter]")

    with pytest.raises(ProofConfigurationError) as error:
        load_universal_token_optimization_proof_config(_write(tmp_path, text))

    assert error.value.reason_code == reason


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    text = _toml() + """

[[cases]]
case_id = "case-one"
source_type = "prompt"
content = "different content"
tags = ["smoke"]

[cases.policy]
enabled = true
profile = "balanced"
allow_lossy = false
require_validation = true
fallback_on_validation_failure = true
"""
    with pytest.raises(ProofConfigurationError, match="DUPLICATE_CASE_IDS"):
        load_universal_token_optimization_proof_config(_write(tmp_path, text))


def test_path_traversal_and_invalid_toml_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProofConfigurationError, match="UNSAFE_OUTPUT_DIRECTORY_PATH"):
        load_universal_token_optimization_proof_config(
            _write(tmp_path, _toml(output="../outside"))
        )

    with pytest.raises(ProofConfigurationError, match="INVALID_TOML"):
        load_universal_token_optimization_proof_config(
            _write(tmp_path, "schema_version = [")
        )


def test_case_path_traversal_and_schema_errors_are_rejected(tmp_path: Path) -> None:
    unsafe_case = _toml().replace('case_id = "case-one"', 'case_id = "../case"')
    with pytest.raises(ProofConfigurationError, match="INVALID_CASE_ID"):
        load_universal_token_optimization_proof_config(_write(tmp_path, unsafe_case))

    unsupported = _toml().replace(
        'schema_version = "token-optimization-proof.v1"',
        'schema_version = "token-optimization-proof.v2"',
    )
    with pytest.raises(ProofConfigurationError, match="UNSUPPORTED_SCHEMA_VERSION"):
        load_universal_token_optimization_proof_config(_write(tmp_path, unsupported))

    with pytest.raises(ProofConfigurationError, match="CONFIG_NOT_FOUND"):
        load_universal_token_optimization_proof_config(tmp_path / "missing.toml")


def test_missing_required_environment_is_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SYNTHETIC_REQUIRED_API_KEY", raising=False)
    text = _toml().replace("run_mode = \"offline_smoke\"", 'run_mode = "live_adapter"')

    with pytest.raises(ProofConfigurationError) as error:
        load_universal_token_optimization_proof_config(_write(tmp_path, text))

    assert error.value.reason_code == "MISSING_API_KEY_ENV"
    assert "secret" not in repr(error.value).lower()
    assert "SYNTHETIC_REQUIRED_API_KEY" not in str(error.value)
