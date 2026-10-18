# © Artur Czarnecki. All rights reserved.

"""Provider-neutral host persistence ports + local implementations (PC-6)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_evidence.receipt import (
    ExecutionEvidenceProofReceipt,
    ProofReceipt,
    parse_execution_evidence_proof_receipt_json,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_execution_result import (
    AnyGovernedExecutionResult,
    GovernedExecutionResult,
    parse_governed_execution_result_json,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationIntentWriteResult,
    ProviderInvocationOutcomeConflictError,
    provider_invocation_outcomes_equivalent,
    provider_invocations_equivalent,
)
from intergrax.contracts.governed_external_work_create_follow_on import (
    GovernedExternalWorkCreateFollowOnRecordV1,
    follow_on_records_equivalent,
    parse_governed_external_work_create_follow_on_record_json,
)
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
)


class GovernedExecutionCreateFollowOnConflictError(ValueError):
    """Conflicting follow-on record for the same execution_id."""


@runtime_checkable
class GovernedExecutionStore(Protocol):
    def put_result(self, result: AnyGovernedExecutionResult) -> None: ...

    def get_result(self, execution_id: str) -> AnyGovernedExecutionResult | None: ...

    def put_state(
        self,
        execution_id: str,
        state: GovernedExternalWorkHostState,
    ) -> None: ...

    def get_state(self, execution_id: str) -> GovernedExternalWorkHostState | None: ...

    def put_event_json(self, execution_id: str, event_json: str) -> None: ...

    def get_event_json(self, execution_id: str) -> str | None: ...

    def put_create_follow_on_record(
        self,
        execution_id: str,
        record: GovernedExternalWorkCreateFollowOnRecordV1,
    ) -> None: ...

    def get_create_follow_on_record(
        self,
        execution_id: str,
    ) -> GovernedExternalWorkCreateFollowOnRecordV1 | None: ...


@runtime_checkable
class ProofReceiptStore(Protocol):
    def put_receipt(self, execution_id: str, receipt: ExecutionEvidenceProofReceipt) -> None: ...

    def get_receipt(self, execution_id: str) -> ExecutionEvidenceProofReceipt | None: ...


@runtime_checkable
class PolicyBundleArtifactStore(Protocol):
    def put_bundle(self, bundle: ImmutableRuntimePolicyBundle) -> None: ...

    def get_bundle(self, bundle_id: str, version: str) -> ImmutableRuntimePolicyBundle | None: ...


@runtime_checkable
class ContinuationStateStore(Protocol):
    def put_continuation(
        self,
        task_id: str,
        continuation: GovernedContinuationRequest,
    ) -> None: ...

    def get_continuation(self, task_id: str) -> GovernedContinuationRequest | None: ...


class InMemoryGovernedExecutionStore:
    @property
    def is_durable(self) -> bool:
        return False

    def __init__(self) -> None:
        self._results: dict[str, AnyGovernedExecutionResult] = {}
        self._states: dict[str, GovernedExternalWorkHostState] = {}
        self._events: dict[str, str] = {}
        self._follow_on: dict[str, GovernedExternalWorkCreateFollowOnRecordV1] = {}

    def put_result(self, result: AnyGovernedExecutionResult) -> None:
        key = str(result.execution_id)
        self._results[key] = result

    def get_result(self, execution_id: str) -> AnyGovernedExecutionResult | None:
        return self._results.get(execution_id)

    def put_state(
        self,
        execution_id: str,
        state: GovernedExternalWorkHostState,
    ) -> None:
        self._states[execution_id] = state

    def get_state(self, execution_id: str) -> GovernedExternalWorkHostState | None:
        return self._states.get(execution_id)

    def put_event_json(self, execution_id: str, event_json: str) -> None:
        self._events[execution_id] = event_json

    def get_event_json(self, execution_id: str) -> str | None:
        return self._events.get(execution_id)

    def put_create_follow_on_record(
        self,
        execution_id: str,
        record: GovernedExternalWorkCreateFollowOnRecordV1,
    ) -> None:
        key = execution_id.strip()
        if str(record.execution_id) != key:
            raise ValueError("follow_on_execution_id_key_mismatch")
        existing = self._follow_on.get(key)
        if existing is not None and not follow_on_records_equivalent(existing, record):
            raise GovernedExecutionCreateFollowOnConflictError(
                f"conflicting_create_follow_on_record:{key}",
            )
        self._follow_on[key] = record

    def get_create_follow_on_record(
        self,
        execution_id: str,
    ) -> GovernedExternalWorkCreateFollowOnRecordV1 | None:
        return self._follow_on.get(execution_id)


class InMemoryProofReceiptStore:
    @property
    def is_durable(self) -> bool:
        return False

    def __init__(self) -> None:
        self._receipts: dict[str, ExecutionEvidenceProofReceipt] = {}

    def put_receipt(self, execution_id: str, receipt: ExecutionEvidenceProofReceipt) -> None:
        self._receipts[execution_id] = receipt

    def get_receipt(self, execution_id: str) -> ExecutionEvidenceProofReceipt | None:
        return self._receipts.get(execution_id)


class InMemoryPolicyBundleArtifactStore:
    @property
    def is_durable(self) -> bool:
        return False

    def __init__(self) -> None:
        self._bundles: dict[tuple[str, str], ImmutableRuntimePolicyBundle] = {}

    def put_bundle(self, bundle: ImmutableRuntimePolicyBundle) -> None:
        self._bundles[(bundle.bundle_id, bundle.version)] = bundle

    def get_bundle(
        self,
        bundle_id: str,
        version: str,
    ) -> ImmutableRuntimePolicyBundle | None:
        return self._bundles.get((bundle_id, version))


class InMemoryProviderInvocationStore:
    """Test/demo fixture — not for production composition."""

    def __init__(self) -> None:
        self._invocations: dict[str, ProviderInvocation] = {}
        self._outcomes: dict[str, ProviderInvocationOutcome] = {}
        self._lock = threading.RLock()

    @property
    def is_durable(self) -> bool:
        return False

    def insert_invocation_if_absent(
        self,
        invocation: ProviderInvocation,
    ) -> ProviderInvocationIntentWriteResult:
        with self._lock:
            existing = self._invocations.get(invocation.invocation_id)
            if existing is not None:
                if provider_invocations_equivalent(existing, invocation):
                    return ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP
                raise ProviderInvocationConflictError(
                    f"provider invocation conflict:{invocation.invocation_id}",
                )
            self._invocations[invocation.invocation_id] = invocation
            return ProviderInvocationIntentWriteResult.INSERTED

    def put_invocation(self, invocation: ProviderInvocation) -> None:
        with self._lock:
            existing = self._invocations.get(invocation.invocation_id)
            if existing is not None:
                if not provider_invocations_equivalent(existing, invocation):
                    raise ProviderInvocationConflictError(
                        f"provider invocation conflict:{invocation.invocation_id}",
                    )
                return
            self._invocations[invocation.invocation_id] = invocation

    def get_invocation(self, invocation_id: str) -> ProviderInvocation | None:
        return self._invocations.get(invocation_id)

    def put_outcome(self, outcome: ProviderInvocationOutcome) -> None:
        existing = self._outcomes.get(outcome.invocation_id)
        if existing is not None:
            if not provider_invocation_outcomes_equivalent(existing, outcome):
                raise ProviderInvocationOutcomeConflictError(
                    f"provider invocation outcome conflict:{outcome.invocation_id}",
                )
            return
        self._outcomes[outcome.invocation_id] = outcome

    def get_outcome(self, invocation_id: str) -> ProviderInvocationOutcome | None:
        return self._outcomes.get(invocation_id)


class InMemoryContinuationStateStore:
    @property
    def is_durable(self) -> bool:
        return False

    def __init__(self) -> None:
        self._items: dict[str, GovernedContinuationRequest] = {}

    def put_continuation(
        self,
        task_id: str,
        continuation: GovernedContinuationRequest,
    ) -> None:
        self._items[task_id] = continuation

    def get_continuation(self, task_id: str) -> GovernedContinuationRequest | None:
        return self._items.get(task_id)


class FilesystemHostStore:
    """Filesystem-backed demo store (no distributed event store)."""

    @property
    def is_durable(self) -> bool:
        return False

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "executions").mkdir(exist_ok=True)
        (self.root / "receipts").mkdir(exist_ok=True)
        (self.root / "bundles").mkdir(exist_ok=True)
        (self.root / "continuations").mkdir(exist_ok=True)
        (self.root / "states").mkdir(exist_ok=True)
        (self.root / "events").mkdir(exist_ok=True)
        (self.root / "create_follow_on").mkdir(exist_ok=True)

    def put_result(self, result: AnyGovernedExecutionResult) -> None:
        key = str(result.execution_id)
        path = self.root / "executions" / f"{key}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def get_result(self, execution_id: str) -> AnyGovernedExecutionResult | None:
        path = self.root / "executions" / f"{execution_id}.json"
        if not path.is_file():
            return None
        try:
            return parse_governed_execution_result_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"corrupted_execution_artifact:{execution_id}") from exc

    def put_state(
        self,
        execution_id: str,
        state: GovernedExternalWorkHostState,
    ) -> None:
        path = self.root / "states" / f"{execution_id}.txt"
        path.write_text(state.value, encoding="utf-8")

    def get_state(self, execution_id: str) -> GovernedExternalWorkHostState | None:
        path = self.root / "states" / f"{execution_id}.txt"
        if not path.is_file():
            return None
        return GovernedExternalWorkHostState(path.read_text(encoding="utf-8").strip())

    def put_event_json(self, execution_id: str, event_json: str) -> None:
        path = self.root / "events" / f"{execution_id}.json"
        path.write_text(event_json, encoding="utf-8")

    def get_event_json(self, execution_id: str) -> str | None:
        path = self.root / "events" / f"{execution_id}.json"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def put_create_follow_on_record(
        self,
        execution_id: str,
        record: GovernedExternalWorkCreateFollowOnRecordV1,
    ) -> None:
        key = execution_id.strip()
        if str(record.execution_id) != key:
            raise ValueError("follow_on_execution_id_key_mismatch")
        path = self.root / "create_follow_on" / f"{key}.json"
        if path.is_file():
            existing = parse_governed_external_work_create_follow_on_record_json(
                path.read_text(encoding="utf-8"),
            )
            if follow_on_records_equivalent(existing, record):
                return
            raise GovernedExecutionCreateFollowOnConflictError(
                f"conflicting_create_follow_on_record:{key}",
            )
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def get_create_follow_on_record(
        self,
        execution_id: str,
    ) -> GovernedExternalWorkCreateFollowOnRecordV1 | None:
        path = self.root / "create_follow_on" / f"{execution_id}.json"
        if not path.is_file():
            return None
        try:
            return parse_governed_external_work_create_follow_on_record_json(
                path.read_text(encoding="utf-8"),
            )
        except ValueError as exc:
            raise ValueError(
                f"corrupted_create_follow_on_artifact:{execution_id}",
            ) from exc

    def put_receipt(self, execution_id: str, receipt: ExecutionEvidenceProofReceipt) -> None:
        path = self.root / "receipts" / f"{execution_id}.json"
        path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")

    def get_receipt(self, execution_id: str) -> ExecutionEvidenceProofReceipt | None:
        path = self.root / "receipts" / f"{execution_id}.json"
        if not path.is_file():
            return None
        try:
            return parse_execution_evidence_proof_receipt_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"corrupted_receipt_artifact:{execution_id}") from exc

    def put_bundle(self, bundle: ImmutableRuntimePolicyBundle) -> None:
        safe = f"{bundle.bundle_id}__{bundle.version}.json".replace("/", "_")
        path = self.root / "bundles" / safe
        path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    def get_bundle(
        self,
        bundle_id: str,
        version: str,
    ) -> ImmutableRuntimePolicyBundle | None:
        safe = f"{bundle_id}__{version}.json".replace("/", "_")
        path = self.root / "bundles" / safe
        if not path.is_file():
            return None
        try:
            return ImmutableRuntimePolicyBundle.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"corrupted_bundle_artifact:{bundle_id}:{version}") from exc

    def put_continuation(
        self,
        task_id: str,
        continuation: GovernedContinuationRequest,
    ) -> None:
        path = self.root / "continuations" / f"{task_id}.json"
        path.write_text(continuation.model_dump_json(indent=2), encoding="utf-8")

    def get_continuation(self, task_id: str) -> GovernedContinuationRequest | None:
        path = self.root / "continuations" / f"{task_id}.json"
        if not path.is_file():
            return None
        return GovernedContinuationRequest.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def write_json(self, relative: str, payload: dict) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
