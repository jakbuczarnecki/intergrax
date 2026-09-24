# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""JSON codec for durable WorkerCapabilityNeed records (UCA-6C-R6-R5.9)."""

from __future__ import annotations

import json
from datetime import datetime

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
    ProfileVersion,
)
from intergrax.contracts.autonomous_work.references import ProblemReference

WORKER_CAPABILITY_NEED_CODEC_VERSION = 1


def _profile_ref_payload(
    ref: CapabilityProfileRef | CodecraftProfileRef,
) -> dict[str, object]:
    return {
        "profile_id": ref.profile_id,
        "version": ref.version.value,
    }


def _profile_ref_from_payload(
    payload: dict[str, object],
    *,
    ref_type: type[CapabilityProfileRef] | type[CodecraftProfileRef],
) -> CapabilityProfileRef | CodecraftProfileRef:
    profile_id = payload.get("profile_id")
    version_raw = payload.get("version")
    if not isinstance(profile_id, str) or not isinstance(version_raw, int):
        raise ValueError("invalid profile reference payload")
    return ref_type(profile_id=profile_id, version=ProfileVersion(version_raw))


def worker_capability_need_to_payload(need: WorkerCapabilityNeed) -> dict[str, object]:
    return {
        "codec_version": WORKER_CAPABILITY_NEED_CODEC_VERSION,
        "worker_instance_id": str(need.worker_instance_id),
        "obstacle_id": need.obstacle_id,
        "need_kind": need.need_kind.value,
        "required_operations": list(need.required_operations),
        "capability_profile_ref": _profile_ref_payload(need.capability_profile_ref),
        "requested_at": need.requested_at.isoformat(),
        "recovery_decision_id": need.recovery_decision_id,
        "evidence_refs": [str(ref) for ref in need.evidence_refs],
        "recovery_episode_id": need.recovery_episode_id,
        "required_data_domains": list(need.required_data_domains),
        "required_protocols": list(need.required_protocols),
        "required_resource_refs": list(need.required_resource_refs),
        "codecraft_profile_ref": (
            _profile_ref_payload(need.codecraft_profile_ref)
            if need.codecraft_profile_ref is not None
            else None
        ),
    }


def worker_capability_need_from_payload(
    payload: dict[str, object],
) -> WorkerCapabilityNeed:
    version = payload.get("codec_version")
    if version != WORKER_CAPABILITY_NEED_CODEC_VERSION:
        raise ValueError("unsupported worker capability need codec version")
    worker_raw = payload.get("worker_instance_id")
    obstacle_id = payload.get("obstacle_id")
    need_kind_raw = payload.get("need_kind")
    operations = payload.get("required_operations")
    profile_payload = payload.get("capability_profile_ref")
    requested_raw = payload.get("requested_at")
    recovery_decision_id = payload.get("recovery_decision_id")
    if (
        not isinstance(worker_raw, str)
        or not isinstance(obstacle_id, str)
        or not isinstance(need_kind_raw, str)
        or not isinstance(operations, list)
        or not isinstance(profile_payload, dict)
        or not isinstance(requested_raw, str)
        or not isinstance(recovery_decision_id, str)
    ):
        raise ValueError("corrupt worker capability need payload")
    evidence_raw = payload.get("evidence_refs", [])
    if not isinstance(evidence_raw, list):
        raise ValueError("corrupt evidence_refs")
    recovery_episode_id = payload.get("recovery_episode_id")
    if recovery_episode_id is not None and not isinstance(recovery_episode_id, str):
        raise ValueError("corrupt recovery_episode_id")
    data_domains = payload.get("required_data_domains", [])
    protocols = payload.get("required_protocols", [])
    resource_refs = payload.get("required_resource_refs", [])
    if not isinstance(data_domains, list) or not isinstance(protocols, list):
        raise ValueError("corrupt need domain/protocol lists")
    if not isinstance(resource_refs, list):
        raise ValueError("corrupt required_resource_refs")
    codecraft_payload = payload.get("codecraft_profile_ref")
    codecraft_ref = None
    if codecraft_payload is not None:
        if not isinstance(codecraft_payload, dict):
            raise ValueError("corrupt codecraft_profile_ref")
        parsed_codecraft = _profile_ref_from_payload(
            codecraft_payload,
            ref_type=CodecraftProfileRef,
        )
        if not isinstance(parsed_codecraft, CodecraftProfileRef):
            raise TypeError("codecraft_profile_ref type mismatch")
        codecraft_ref = parsed_codecraft
    capability_profile_ref = _profile_ref_from_payload(
        profile_payload,
        ref_type=CapabilityProfileRef,
    )
    if not isinstance(capability_profile_ref, CapabilityProfileRef):
        raise TypeError("capability_profile_ref type mismatch")
    return WorkerCapabilityNeed(
        worker_instance_id=WorkerInstanceId(worker_raw),
        obstacle_id=obstacle_id,
        need_kind=CapabilityNeedKind(need_kind_raw),
        required_operations=tuple(str(op) for op in operations),
        capability_profile_ref=capability_profile_ref,
        requested_at=datetime.fromisoformat(requested_raw),
        recovery_decision_id=recovery_decision_id,
        evidence_refs=tuple(ProblemReference(str(ref)) for ref in evidence_raw),
        recovery_episode_id=recovery_episode_id,
        required_data_domains=tuple(str(d) for d in data_domains),
        required_protocols=tuple(str(p) for p in protocols),
        required_resource_refs=tuple(str(r) for r in resource_refs),
        codecraft_profile_ref=codecraft_ref,
    )


def worker_capability_need_to_json(need: WorkerCapabilityNeed) -> str:
    return json.dumps(worker_capability_need_to_payload(need), sort_keys=True)


def worker_capability_need_from_json(record_json: str) -> WorkerCapabilityNeed:
    payload = json.loads(record_json)
    if not isinstance(payload, dict):
        raise ValueError("worker capability need record must be a JSON object")
    return worker_capability_need_from_payload(payload)


__all__ = [
    "WORKER_CAPABILITY_NEED_CODEC_VERSION",
    "worker_capability_need_from_json",
    "worker_capability_need_from_payload",
    "worker_capability_need_to_json",
    "worker_capability_need_to_payload",
]
