from __future__ import annotations

from intergrax.platform.enterprise_audit.sandbox import _redact


def test_redact_strips_api_key_and_uuid_lines() -> None:
    raw = (
        "trace id 550e8400-e29b-41d4-a716-446655440000\n"
        "export API_KEY=sk-live-not-a-real-key\n"
        "PASSWORD=hunter2\n"
        "benign log line\n"
    )
    redacted = _redact(raw)
    assert "550e8400" not in redacted
    assert "sk-live" not in redacted
    assert "hunter2" not in redacted
    assert "benign log line" in redacted
    assert redacted.count("[REDACTED]") >= 2
