"""Red-first NT-RPL probes for MessageConsumer vendor replaceability."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CONSUMER_WORKERS = (
    _REPO / "intergrax/queueing/providers/kafka/kafka_worker.py",
    _REPO / "intergrax/queueing/providers/rabbitmq/rabbitmq_worker.py",
)
_FORBIDDEN_CONCRETE = frozenset({"ConfluentKafkaMessageConsumer", "RabbitMQMessageConsumer"})


def _imports_concrete_message_consumer(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _FORBIDDEN_CONCRETE:
                    found.append(alias.name)
    return found


def test_neg_p_rpl_04_queue_worker_call_sites_bind_message_consumer_contract() -> None:
    """NEG-P / FRZ-RPL-04: vendor/backend switch must not require editing consumer call sites."""
    for path in _CONSUMER_WORKERS:
        rel = path.relative_to(_REPO).as_posix()
        concrete_imports = _imports_concrete_message_consumer(path)
        assert not concrete_imports, (
            "NEG-P vendor/backend switch does not modify consumers: "
            f"{rel} must not import a concrete MessageConsumer vendor class "
            f"({', '.join(concrete_imports)})"
        )
        text = path.read_text(encoding="utf-8")
        assert "MessageConsumer" in text, (
            "NEG-P vendor/backend switch does not modify consumers: "
            f"{rel} must type the polling surface as MessageConsumer"
        )
        assert "consumer:MessageConsumer" in text.replace(" ", ""), (
            "NEG-P vendor/backend switch does not modify consumers: "
            f"{rel} constructor must accept MessageConsumer, not a Kafka/RabbitMQ concrete"
        )
