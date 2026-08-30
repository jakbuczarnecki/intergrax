from __future__ import annotations

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.observability.modality_metrics import extract_modality_metrics


def test_extract_modality_metrics_from_event_payload() -> None:
    event = RuntimeEvent(
        phase=ExecutionPhase.COMPLETION,
        event_type=RuntimeEventType.TASK_COMPLETED,
        tenant_id="tenant-a",
        agent_id="agent:echo",
        payload={
            "modality_metrics": {
                "inference_ms": 120,
                "media_bytes": 4096,
                "tts_characters": 42,
            }
        },
        **runtime_event_test_identity(),
    )
    metrics = extract_modality_metrics(event)
    assert metrics.inference_ms == 120
    assert metrics.media_bytes == 4096
    assert metrics.tts_characters == 42
