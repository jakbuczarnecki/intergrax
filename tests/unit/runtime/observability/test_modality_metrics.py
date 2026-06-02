from __future__ import annotations

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.modality_metrics import extract_modality_metrics


def test_extract_modality_metrics_from_event_payload() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        phase=ExecutionPhase.COMPLETION,
        event_type=RuntimeEventType.TASK_COMPLETED,
        run_id="run-1",
        tenant_id="tenant-a",
        agent_id="agent:echo",
        payload={
            "modality_metrics": {
                "inference_ms": 120,
                "media_bytes": 4096,
                "tts_characters": 42,
            }
        },
    )
    metrics = extract_modality_metrics(event)
    assert metrics.inference_ms == 120
    assert metrics.media_bytes == 4096
    assert metrics.tts_characters == 42
