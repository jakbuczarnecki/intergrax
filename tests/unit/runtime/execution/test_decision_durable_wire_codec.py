# © Artur Czarnecki. All rights reserved.

"""DS-REC wire codec — safe durable serialization boundary."""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
from intergrax.contracts.decision_finalization import (
    decision_finalization_key,
    guard_decision_finalization,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionProposalRef,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.decision_revision import decision_revision_checkpoint_state
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    decision_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.decision_durable_wire_codec import (
    decode_checkpoint_blob,
    decode_outcome_blob,
    encode_checkpoint_blob,
    encode_outcome_blob,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.decision_persistence_codec_errors import (
    DecisionPersistenceCodecError,
    DecisionPersistenceLegacyPickleUnsupportedError,
    DecisionPersistenceRecordTypeError,
    DecisionPersistenceUnsupportedSchemaError,
    DecisionPersistenceUnknownPayloadCodecError,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _payload_registry() -> object:
    return conformance_artifact_payload_codec_registry()


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _accepted(identity: DecisionIdentity) -> AuthoritativeAcceptedDecision[IncidentDecisionPayload]:
    return AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="escalate"),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )


def _checkpoint_with_revision(identity: DecisionIdentity) -> object:
    lifecycle = transition_decision_lifecycle(
        transition_decision_lifecycle(
            initial_decision_lifecycle_state(identity),
            DecisionLifecycleStage.VERIFICATION,
        ),
        DecisionLifecycleStage.REVISION,
    )
    revision = decision_revision_checkpoint_state(
        proposal_ref=DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(identity.version),
        ),
        revision_count=2,
        max_revisions=3,
    )
    return decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )


def test_identity_round_trip() -> None:
    identity = _identity()
    checkpoint = decision_checkpoint_state(
        lifecycle=initial_decision_lifecycle_state(identity),
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
    )
    blob = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    restored = decode_checkpoint_blob(blob, payload_codecs=_payload_registry())
    assert restored.lifecycle.identity == identity


def test_lifecycle_round_trip() -> None:
    identity = _identity()
    lifecycle = transition_decision_lifecycle(
        initial_decision_lifecycle_state(identity),
        DecisionLifecycleStage.DELIBERATION,
    )
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
    )
    blob = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    restored = decode_checkpoint_blob(blob, payload_codecs=_payload_registry())
    assert restored.lifecycle == lifecycle


def test_revision_checkpoint_round_trip() -> None:
    identity = _identity()
    checkpoint = _checkpoint_with_revision(identity)
    blob = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    restored = decode_checkpoint_blob(blob, payload_codecs=_payload_registry())
    assert restored.revision == checkpoint.revision


def test_checkpoint_v2_round_trip() -> None:
    identity = _identity()
    checkpoint = _checkpoint_with_revision(identity)
    blob = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    restored = decode_checkpoint_blob(blob, payload_codecs=_payload_registry())
    assert restored == checkpoint


def test_accepted_decision_round_trip() -> None:
    identity = _identity()
    accepted = _accepted(identity)
    blob = encode_outcome_blob(accepted, payload_codecs=_payload_registry())
    restored = decode_outcome_blob(blob, payload_codecs=_payload_registry())
    assert restored == accepted


def test_resolution_round_trip_rejected_and_unresolved() -> None:
    identity = _identity()
    for resolution in (DecisionResolution.REJECTED, DecisionResolution.UNRESOLVED):
        record = AuthoritativeResolutionRecord(identity=identity, resolution=resolution)
        blob = encode_outcome_blob(record, payload_codecs=_payload_registry())
        restored = decode_outcome_blob(blob, payload_codecs=_payload_registry())
        assert restored == record


def test_lineage_with_parents_round_trip() -> None:
    base_identity = _identity()
    identity_v2 = DecisionIdentity(
        decision_id=base_identity.decision_id,
        version=next_decision_version(initial_decision_version()),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=base_identity.execution,
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity_v2.version),
        parents=(decision_lineage_ref(initial_decision_version()),),
    )
    accepted = AuthoritativeAcceptedDecision(
        identity=identity_v2,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="contain"),
        ),
        lineage=lineage,
    )
    blob = encode_outcome_blob(accepted, payload_codecs=_payload_registry())
    restored = decode_outcome_blob(blob, payload_codecs=_payload_registry())
    assert restored == accepted


def test_encode_is_deterministic() -> None:
    identity = _identity()
    checkpoint = _checkpoint_with_revision(identity)
    first = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    second = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    assert first == second


def test_legacy_pickle_blob_rejected_without_execution() -> None:
    identity = _identity()
    checkpoint = decision_checkpoint_state(
        lifecycle=initial_decision_lifecycle_state(identity),
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
    )
    legacy_blob = base64.b64encode(
        pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL),
    ).decode("ascii")
    with pytest.raises(DecisionPersistenceLegacyPickleUnsupportedError):
        decode_checkpoint_blob(legacy_blob, payload_codecs=_payload_registry())


@pytest.mark.parametrize(
    ("blob", "error_type"),
    [
        ("", DecisionPersistenceCodecError),
        ("{", DecisionPersistenceCodecError),
        (
            '{"schema_version":999,"record_type":"decision_checkpoint","payload":{}}',
            DecisionPersistenceUnsupportedSchemaError,
        ),
        (
            '{"schema_version":2,"record_type":"unknown","payload":{}}',
            DecisionPersistenceRecordTypeError,
        ),
        (
            '{"schema_version":2,"record_type":"decision_checkpoint","payload":{}}',
            DecisionPersistenceCodecError,
        ),
    ],
)
def test_corruption_matrix(blob: str, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        decode_checkpoint_blob(blob, payload_codecs=_payload_registry())


def test_unknown_payload_codec_fails_closed_on_decode() -> None:
    identity = _identity()
    accepted = _accepted(identity)
    blob = encode_outcome_blob(accepted, payload_codecs=_payload_registry())
    empty_registry = decision_artifact_payload_codec_registry(codecs={})
    with pytest.raises(DecisionPersistenceUnknownPayloadCodecError):
        decode_outcome_blob(blob, payload_codecs=empty_registry)


def test_unknown_payload_codec_fails_closed_on_encode() -> None:
    identity = _identity()
    accepted = _accepted(identity)
    empty_registry = decision_artifact_payload_codec_registry(codecs={})
    with pytest.raises(DecisionPersistenceUnknownPayloadCodecError):
        encode_outcome_blob(accepted, payload_codecs=empty_registry)


def test_decision_durable_persistence_production_files_contain_no_pickle() -> None:
    repo_root = Path("intergrax/runtime/execution")
    targets = sorted(repo_root.glob("*decision*persistence*.py")) + sorted(
        repo_root.glob("*decision*codec*.py"),
    )
    targets = [
        path
        for path in targets
        if path.name
        not in {
            "decision_persistence_codec_errors.py",
        }
    ]
    assert targets
    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert "pickle" not in source, f"forbidden pickle reference in {path}"


def test_codec_production_modules_contain_no_reflection() -> None:
    forbidden = ("getattr(", "setattr(", "hasattr(", "importlib", "__import__", "eval(", "exec(")
    targets = [
        Path("intergrax/runtime/execution/decision_durable_wire_codec.py"),
        Path("intergrax/runtime/execution/decision_artifact_payload_codec.py"),
        Path("intergrax/runtime/execution/decision_persistence_codec_errors.py"),
    ]
    for path in targets:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"forbidden {token!r} in {path}"


def test_finalization_guard_with_accepted_round_trip_in_checkpoint() -> None:
    identity = _identity()
    accepted = _accepted(identity)
    guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        accepted,
    ).state
    lifecycle = transition_decision_lifecycle(
        transition_decision_lifecycle(
            transition_decision_lifecycle(
                initial_decision_lifecycle_state(identity),
                DecisionLifecycleStage.VERIFICATION,
            ),
            DecisionLifecycleStage.RESOLUTION,
        ),
        DecisionLifecycleStage.FINALIZATION,
    )
    checkpoint = decision_checkpoint_state(lifecycle=lifecycle, finalization=guard)
    blob = encode_checkpoint_blob(checkpoint, payload_codecs=_payload_registry())
    restored = decode_checkpoint_blob(blob, payload_codecs=_payload_registry())
    assert restored.finalization.authoritative_outcome == accepted


@dataclass(frozen=True, slots=True)
class _UnknownPayload:
    value: str


def test_unknown_artifact_kind_codec_on_checkpoint_decode() -> None:
    identity = _identity()
    unknown_accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("unknown_kind"),
            content=_UnknownPayload(value="x"),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        unknown_accepted,
    ).state
    lifecycle = transition_decision_lifecycle(
        transition_decision_lifecycle(
            transition_decision_lifecycle(
                initial_decision_lifecycle_state(identity),
                DecisionLifecycleStage.VERIFICATION,
            ),
            DecisionLifecycleStage.RESOLUTION,
        ),
        DecisionLifecycleStage.FINALIZATION,
    )
    checkpoint = decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=guard,
    )
    kind = validate_decision_artifact_kind("unknown_kind")

    @dataclass(frozen=True, slots=True)
    class _UnknownCodec:
        def encode(self, payload: _UnknownPayload) -> dict[str, str]:
            return {"value": payload.value}

        def decode(self, payload: object) -> _UnknownPayload:
            if type(payload) is not dict:
                raise TypeError("payload must be dict")
            value = payload.get("value")
            if type(value) is not str:
                raise TypeError("value must be str")
            return _UnknownPayload(value=value)

    encode_registry = decision_artifact_payload_codec_registry(
        codecs={kind: _UnknownCodec()},
    )
    blob = encode_checkpoint_blob(checkpoint, payload_codecs=encode_registry)
    with pytest.raises(DecisionPersistenceUnknownPayloadCodecError):
        decode_checkpoint_blob(blob, payload_codecs=_payload_registry())
