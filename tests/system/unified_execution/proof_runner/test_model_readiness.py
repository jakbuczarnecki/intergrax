# © Artur Czarnecki. All rights reserved.

"""Unit tests for certification model readiness bootstrap."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.model_readiness import (
    ModelCapability,
    ModelReadinessError,
    ModelReadinessProbeConfig,
    ModelRequirement,
    OllamaHttpClient,
    OllamaReadinessAdapter,
    _probe_ollama_embedding,
    _probe_ollama_generation,
    _sorted_requirements,
    _valid_embedding_vector,
    c1_model_requirements,
    ensure_model_readiness,
)

pytestmark = pytest.mark.unit

_CONFIG = ProofConfig(
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
    llm_model="llama3.1:latest",
)


def _tags_payload(model_name: str) -> dict[str, object]:
    return {
        "models": [
            {
                "name": model_name,
                "digest": "sha256:abc123",
            }
        ]
    }


def test_valid_embedding_vector_accepts_finite_numeric_vector() -> None:
    assert _valid_embedding_vector([0.1, -0.2, 1.0])


def test_valid_embedding_vector_rejects_empty_and_malformed() -> None:
    assert not _valid_embedding_vector([])
    assert not _valid_embedding_vector(["bad"])
    assert not _valid_embedding_vector([float("inf")])


def test_presence_in_tags_without_inference_probe_is_not_ready() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.get_json.return_value = _tags_payload("nomic-embed-text")
    client.post_json.return_value = {"embedding": []}
    adapter = OllamaReadinessAdapter(
        base_url="http://ollama:11434",
        client=client,
        pull_timeout_seconds=60.0,
    )

    assert adapter.ensure_present("nomic-embed-text")
    ready, error_code = adapter.probe_capability(
        "nomic-embed-text",
        ModelCapability.EMBEDDING,
    )
    assert not ready
    assert error_code == "embedding_vector_invalid"


def test_embedding_inference_returns_valid_vector_is_ready() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.post_json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    ready, error_code = _probe_ollama_embedding(client, model_id="nomic-embed-text")
    assert ready
    assert error_code is None


def test_malformed_embedding_response_is_not_ready() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.post_json.return_value = {"embedding": []}
    ready, error_code = _probe_ollama_embedding(client, model_id="nomic-embed-text")
    assert not ready
    assert error_code == "embedding_vector_invalid"

    client.post_json.return_value = {"unexpected": True}
    ready, error_code = _probe_ollama_embedding(client, model_id="nomic-embed-text")
    assert not ready
    assert error_code == "embedding_vector_invalid"


def test_generation_probe_requires_non_empty_response() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.post_json.return_value = {"response": "x"}
    ready, error_code = _probe_ollama_generation(client, model_id="llama3.1:latest")
    assert ready
    assert error_code is None

    client.post_json.return_value = {"response": ""}
    ready, error_code = _probe_ollama_generation(client, model_id="llama3.1:latest")
    assert not ready
    assert error_code == "generation_response_invalid"


def test_retry_succeeds_on_third_attempt() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.get_json.return_value = _tags_payload("nomic-embed-text")
    client.post_json.side_effect = [
        {"embedding": []},
        {"embedding": []},
        {"embedding": [0.1, 0.2]},
    ]

    with patch(
        "tests.system.unified_execution.proof_runner.model_readiness.OllamaHttpClient",
        return_value=client,
    ):
        with patch(
            "tests.system.unified_execution.proof_runner.model_readiness.time.sleep",
        ) as sleep_mock:
            report = ensure_model_readiness(
                _CONFIG,
                [
                    ModelRequirement(
                        provider="ollama",
                        model_id="nomic-embed-text",
                        capability=ModelCapability.EMBEDDING,
                    )
                ],
                probe_config=ModelReadinessProbeConfig(
                    max_attempts=3,
                    request_timeout_seconds=5.0,
                    backoff_seconds=0.0,
                ),
            )

    assert len(report.results) == 1
    result = report.results[0]
    assert result.ready
    assert result.attempts == 3
    assert sleep_mock.call_count == 2


def test_exhaustion_raises_model_not_ready() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.get_json.return_value = _tags_payload("nomic-embed-text")
    client.post_json.return_value = {"embedding": []}

    with patch(
        "tests.system.unified_execution.proof_runner.model_readiness.OllamaHttpClient",
        return_value=client,
    ):
        with patch(
            "tests.system.unified_execution.proof_runner.model_readiness.time.sleep",
        ):
            with pytest.raises(ModelReadinessError) as exc_info:
                ensure_model_readiness(
                    _CONFIG,
                    [
                        ModelRequirement(
                            provider="ollama",
                            model_id="nomic-embed-text",
                            capability=ModelCapability.EMBEDDING,
                        )
                    ],
                    probe_config=ModelReadinessProbeConfig(
                        max_attempts=2,
                        request_timeout_seconds=5.0,
                        backoff_seconds=0.0,
                    ),
                )

    message = str(exc_info.value)
    assert message.startswith("MODEL_NOT_READY")
    assert "provider=ollama" in message
    assert "capability=EMBEDDING" in message
    assert "attempts=2" in message
    assert "intergrax certification" not in message.lower()


def test_multiple_requirements_all_must_be_ready() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.get_json.side_effect = [
        _tags_payload("nomic-embed-text"),
        _tags_payload("llama3.1:latest"),
    ]
    client.post_json.side_effect = [
        {"embedding": [0.1]},
        {"response": ""},
    ]

    with patch(
        "tests.system.unified_execution.proof_runner.model_readiness.OllamaHttpClient",
        return_value=client,
    ):
        with patch(
            "tests.system.unified_execution.proof_runner.model_readiness.time.sleep",
        ):
            with pytest.raises(ModelReadinessError):
                ensure_model_readiness(
                    _CONFIG,
                    [
                        ModelRequirement(
                            provider="ollama",
                            model_id="nomic-embed-text",
                            capability=ModelCapability.EMBEDDING,
                        ),
                        ModelRequirement(
                            provider="ollama",
                            model_id="llama3.1:latest",
                            capability=ModelCapability.GENERATION,
                        ),
                    ],
                    probe_config=ModelReadinessProbeConfig(
                        max_attempts=1,
                        request_timeout_seconds=5.0,
                        backoff_seconds=0.0,
                    ),
                )


def test_c1_requirements_exclude_unused_generation_model() -> None:
    requirements = c1_model_requirements(_CONFIG)
    assert len(requirements) == 1
    assert requirements[0].capability is ModelCapability.EMBEDDING
    assert requirements[0].model_id == "nomic-embed-text"
    assert all(item.capability is not ModelCapability.GENERATION for item in requirements)


def test_unused_generation_model_not_in_c1_requirements_blocks_nothing() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.get_json.return_value = _tags_payload("nomic-embed-text")
    client.post_json.return_value = {"embedding": [0.2, 0.3]}

    with patch(
        "tests.system.unified_execution.proof_runner.model_readiness.OllamaHttpClient",
        return_value=client,
    ):
        report = ensure_model_readiness(
            _CONFIG,
            c1_model_requirements(_CONFIG),
            probe_config=ModelReadinessProbeConfig(
                max_attempts=1,
                request_timeout_seconds=5.0,
                backoff_seconds=0.0,
            ),
        )

    assert report.results[0].ready
    assert client.post_json.call_count == 1
    posted = client.post_json.call_args_list[0].args[1]
    assert posted["model"] == "nomic-embed-text"
    assert "/api/embeddings" in client.post_json.call_args_list[0].args[0]


def test_result_projection_contains_bounded_error_fields_only() -> None:
    client = MagicMock(spec=OllamaHttpClient)
    client.get_json.return_value = _tags_payload("nomic-embed-text")
    client.post_json.side_effect = json.JSONDecodeError("bad", "doc", 0)

    with patch(
        "tests.system.unified_execution.proof_runner.model_readiness.OllamaHttpClient",
        return_value=client,
    ):
        with patch(
            "tests.system.unified_execution.proof_runner.model_readiness.time.sleep",
        ):
            with pytest.raises(ModelReadinessError) as exc_info:
                ensure_model_readiness(
                    _CONFIG,
                    c1_model_requirements(_CONFIG),
                    probe_config=ModelReadinessProbeConfig(
                        max_attempts=1,
                        request_timeout_seconds=5.0,
                        backoff_seconds=0.0,
                    ),
                )

    payload = str(exc_info.value)
    assert "json_decode_error" in payload
    assert "doc" not in payload


def test_sorted_requirements_produce_deterministic_probe_order() -> None:
    unordered = [
        ModelRequirement(
            provider="ollama",
            model_id="z-model",
            capability=ModelCapability.GENERATION,
        ),
        ModelRequirement(
            provider="ollama",
            model_id="a-model",
            capability=ModelCapability.EMBEDDING,
        ),
        ModelRequirement(
            provider="ollama",
            model_id="m-model",
            capability=ModelCapability.EMBEDDING,
        ),
    ]
    ordered = _sorted_requirements(unordered)
    assert [item.model_id for item in ordered] == ["a-model", "m-model", "z-model"]
