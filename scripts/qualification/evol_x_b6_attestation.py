# © Artur Czarnecki. All rights reserved.

"""EVOL-X automated B6 attestation contract (evaluation only).

This module verifies signed attestation records and governance status.
It does not mint production signatures, store private keys, or publish
``B6_ACCEPTANCE_PUBLICATION_SHA``.

``sign_attestation`` accepts only ``NON_PRODUCTION_TEST_ONLY`` credentials and
only a ``NOT_ACCEPTED`` disposition. A signature produced here is not
independent custody and cannot authorize formal B6.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

SCHEMA_VERSION = "evol_x_b6_attestation.v1"
TRUST_ROOT_SCHEMA = "evol_x_b6_trust_root.v1"
GOVERNANCE_SCHEMA = "evol_x_auto_b6_governance_status.v1"
POLICY_ID = "ECAP-P4-EVOL-SCOPE-AUTO-B6"
POLICY_VERSION = "2026-11-02.1-pending"
OPERATOR_DECISION = "OD-EVOL-14"
PROFILE = "AUTOMATION_FIRST_SCOPED"
TOOL_ID = "scripts/qualification/evol_x_b6_independent_verification.py"
TOOL_VERSION = "2"
PINNED_TES = "3c4a641b3e48a13f437e830a0562e2a80b03aabb"
PINNED_FINAL = "bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1"
QUALIFIED_CI_RUN_ID = "41337077334"
FINDINGS_REGISTER_ID = "EVOL_X_B6_FINDINGS_REGISTER"
CREDENTIAL_TEST = "NON_PRODUCTION_TEST_ONLY"
CREDENTIAL_PRODUCTION = "PRODUCTION_INDEPENDENT"
CUSTODY_INDEPENDENT = "INDEPENDENT_EXTERNAL"
SIGNATURE_ALGORITHM = "Ed25519"

IMPLEMENTER_IDENTITIES = frozenset(
    {
        "implementer",
        "implementation-agent",
        "cursor-agent",
        "cursor-implementer",
        "self",
    }
)

_PENDING_AUTHORITY_REASONS = frozenset(
    {
        "MISSING_ECAP_MAINTAINER_PUBLICATION",
        "MISSING_SECURITY_REVIEW",
        "MISSING_INDEPENDENT_KEY_CUSTODY",
        "ATTESTATION_ENVIRONMENT_NOT_PROVISIONED",
        "POLICY_NOT_IN_FORCE",
    }
)

_ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "ecap_policy_id",
        "ecap_policy_version",
        "operator_decision_ref",
        "certification_profile",
        "test_execution_sha",
        "evidence_final_publication_sha",
        "verification_tool_id",
        "verification_tool_version",
        "verification_tool_digest",
        "ci_provenance",
        "mode_a_result",
        "mode_b_result",
        "findings_disposition",
        "frz_scope_declaration",
        "attestor_identity",
        "evidence_producer_identity",
        "key_id",
        "trust_root_ref",
        "signature_algorithm",
        "signature",
        "signing_timestamp",
        "validity_not_before",
        "validity_not_after",
        "attestation_id",
        "replay_binding",
        "credential_class",
        "formal_acceptance_disposition",
    }
)
_CI_KEYS = frozenset({"run_id", "conclusion", "head_sha", "profile"})
_MODE_KEYS = frozenset({"status", "m09a", "m09h"})
_FINDINGS_KEYS = frozenset({"register_id", "complete", "open_blocking", "m09h_disposition", "m09a_disposition"})
_FRZ_KEYS = frozenset({"scope", "promotes_frz", "frz_evol_07", "declares_evol_x_closed"})
_TRUST_ROOT_KEYS = frozenset({"schema_version", "trust_root_id", "custody_domain", "keys"})
_TRUST_KEY_KEYS = frozenset(
    {
        "key_id",
        "public_key_b64",
        "status",
        "credential_class",
        "attestor_identity",
        "controlled_by_implementer",
    }
)


class ProductionSigningForbidden(PermissionError):
    """Raised when this workspace is asked to mint a formal or production attestation."""


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def default_governance_path(repo: Path) -> Path:
    return repo / "docs/project/maintainers/qualification/evol_x/EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json"


def load_governance(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("governance status must be a JSON object")
    return data


def sign_attestation(private_key: Ed25519PrivateKey, body: Mapping[str, Any]) -> dict[str, Any]:
    """Sign a disposable test record. Refuses production class and formal acceptance."""
    unsigned = dict(body)
    unsigned.pop("signature", None)
    if unsigned.get("credential_class") != CREDENTIAL_TEST:
        raise ProductionSigningForbidden("refusing to mint a production attestation in the implementer workspace")
    if unsigned.get("formal_acceptance_disposition") != "NOT_ACCEPTED":
        raise ProductionSigningForbidden("test signer cannot emit a formal acceptance disposition")
    unsigned.pop("replay_binding", None)
    unsigned["replay_binding"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    signature = private_key.sign(canonical_bytes(unsigned))
    unsigned["signature"] = base64.b64encode(signature).decode("ascii")
    return unsigned


def build_unsigned_attestation_payload(
    *,
    verification_tool_digest: str,
    attestor_identity: str,
    evidence_producer_identity: str,
    key_id: str,
    trust_root_ref: str,
    attestation_id: str,
    credential_class: str,
    formal_acceptance_disposition: str,
    mode_a_status: str,
    mode_b_status: str,
    now: datetime,
    validity_hours: int = 168,
) -> dict[str, Any]:
    """Build attestation fields without signature or replay_binding (custodian signs externally)."""
    if credential_class not in {CREDENTIAL_TEST, CREDENTIAL_PRODUCTION}:
        raise ValueError("invalid credential_class")
    if formal_acceptance_disposition not in {"NOT_ACCEPTED", "ACCEPTED_AUTOMATED_B6"}:
        raise ValueError("invalid formal_acceptance_disposition")
    from datetime import timedelta

    mode_a = {"status": mode_a_status, "m09a": "MEASURED_PASS", "m09h": "NOT_MEASURED"}
    mode_b = {"status": mode_b_status, "m09a": "MEASURED_PASS", "m09h": "NOT_MEASURED"}
    not_before = now.replace(microsecond=0)
    not_after = not_before + timedelta(hours=validity_hours)
    return {
        "schema_version": SCHEMA_VERSION,
        "ecap_policy_id": POLICY_ID,
        "ecap_policy_version": POLICY_VERSION,
        "operator_decision_ref": OPERATOR_DECISION,
        "certification_profile": PROFILE,
        "test_execution_sha": PINNED_TES,
        "evidence_final_publication_sha": PINNED_FINAL,
        "verification_tool_id": TOOL_ID,
        "verification_tool_version": TOOL_VERSION,
        "verification_tool_digest": verification_tool_digest,
        "ci_provenance": {
            "run_id": QUALIFIED_CI_RUN_ID,
            "conclusion": "success",
            "head_sha": PINNED_TES,
            "profile": "evol_x_p3_l3",
        },
        "mode_a_result": dict(mode_a),
        "mode_b_result": dict(mode_b),
        "findings_disposition": {
            "register_id": FINDINGS_REGISTER_ID,
            "complete": True,
            "open_blocking": 0,
            "m09h_disposition": "NOT_MEASURED",
            "m09a_disposition": "MEASURED_PASS",
        },
        "frz_scope_declaration": {
            "scope": "EVOL-X-ONLY",
            "promotes_frz": False,
            "frz_evol_07": "OPEN_NOT_PROMOTED",
            "declares_evol_x_closed": False,
        },
        "attestor_identity": attestor_identity,
        "evidence_producer_identity": evidence_producer_identity,
        "key_id": key_id,
        "trust_root_ref": trust_root_ref,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "signing_timestamp": now.replace(microsecond=0).isoformat(),
        "validity_not_before": not_before.isoformat(),
        "validity_not_after": not_after.isoformat(),
        "attestation_id": attestation_id,
        "replay_binding": "",
        "credential_class": credential_class,
        "formal_acceptance_disposition": formal_acceptance_disposition,
    }


def bind_and_sign_attestation(private_key: Ed25519PrivateKey, body: Mapping[str, Any]) -> dict[str, Any]:
    """Compute replay_binding and Ed25519 signature (custodian/runtime use only)."""
    unsigned = dict(body)
    unsigned.pop("signature", None)
    unsigned.pop("replay_binding", None)
    unsigned["replay_binding"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    signature = private_key.sign(canonical_bytes(unsigned))
    unsigned["signature"] = base64.b64encode(signature).decode("ascii")
    return unsigned


def evaluate_acceptance(
    *,
    technical_verdict: str,
    mode_a: str,
    mode_b: str,
    governance: Mapping[str, Any],
    attestation: Mapping[str, Any] | None,
    trust_root: Mapping[str, Any] | None,
    observed: Mapping[str, str] | None,
    expected_tool_digest: str | None,
    replay_ledger: set[str] | None,
    now: datetime,
) -> dict[str, Any]:
    if technical_verdict not in {"PASS", "FAIL", "INCOMPLETE", "NOT_RUN"}:
        technical_verdict = "FAIL"
    authority = _authority(governance)
    if attestation is None:
        attestation_result: dict[str, Any] = {
            "verdict": "ABSENT",
            "reasons": ["ATTESTATION_ABSENT"],
            "limitations": ["FORMAL_ATTESTATION_DEPLOYMENT_BLOCKED"],
            "credential_class": None,
        }
    else:
        attestation_result = _evaluate_attestation(
            attestation,
            trust_root=trust_root,
            observed=observed,
            expected_tool_digest=expected_tool_digest,
            replay_ledger=replay_ledger,
            now=now,
        )
    blockers: list[str] = []
    if technical_verdict != "PASS":
        blockers.append("TECHNICAL_VERIFICATION_NOT_PASS")
    if authority["verdict"] != "SATISFIED":
        blockers.append("AUTHORITY_NOT_SATISFIED")
    if attestation_result["verdict"] != "VALID_PRODUCTION_INDEPENDENT":
        blockers.append("ATTESTATION_NOT_PRODUCTION_VALID")
    if attestation_result.get("credential_class") == CREDENTIAL_TEST:
        blockers.append("NON_PRODUCTION_TEST_ONLY")
    return {
        "technical_verification": {
            "verdict": technical_verdict,
            "mode_a": mode_a,
            "mode_b": mode_b,
        },
        "authority_verification": authority,
        "attestation_verification": attestation_result,
        "formal_acceptance_eligibility": "ELIGIBLE" if not blockers else "NOT_ELIGIBLE",
        "formal_acceptance_blockers": blockers,
    }


def _authority(governance: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if governance.get("schema_version") != GOVERNANCE_SCHEMA:
        reasons.append("GOVERNANCE_SCHEMA_INVALID")
    if governance.get("operator_decision_id") != OPERATOR_DECISION:
        reasons.append("INSUFFICIENT_AUTHORITY")
    if governance.get("ecap_amendment_id") != POLICY_ID:
        reasons.append("CONFLICTING_POLICY_RECORD")
    if governance.get("ecap_amendment_version") != POLICY_VERSION:
        reasons.append("CONFLICTING_POLICY_RECORD")
    if governance.get("test_execution_sha") != PINNED_TES:
        reasons.append("CONFLICTING_POLICY_RECORD")
    if governance.get("evidence_final_publication_sha") != PINNED_FINAL:
        reasons.append("CONFLICTING_POLICY_RECORD")
    approvals = governance.get("required_approvals_present")
    flag_codes = {
        "operator_direction": "MISSING_OPERATOR_DIRECTION",
        "ecap_maintainer_publication": "MISSING_ECAP_MAINTAINER_PUBLICATION",
        "security_review": "MISSING_SECURITY_REVIEW",
        "independent_key_custody": "MISSING_INDEPENDENT_KEY_CUSTODY",
    }
    if not isinstance(approvals, dict):
        reasons.append("INSUFFICIENT_AUTHORITY")
    else:
        for flag, code in flag_codes.items():
            if approvals.get(flag) is not True:
                reasons.append(code)
    if governance.get("independent_attestation_environment") is not True:
        reasons.append("ATTESTATION_ENVIRONMENT_NOT_PROVISIONED")
    activation = governance.get("ecap_amendment_activation")
    effect = governance.get("normative_effect")
    if activation != "IN_FORCE" or effect != "IN_FORCE":
        reasons.append("POLICY_NOT_IN_FORCE")
    deduped = list(dict.fromkeys(reasons))
    if not deduped:
        verdict = "SATISFIED"
    elif set(deduped) <= _PENDING_AUTHORITY_REASONS and activation == "PENDING_REQUIRED_APPROVALS":
        verdict = "PENDING_REQUIRED_APPROVALS"
    else:
        verdict = "INSUFFICIENT"
    return {
        "verdict": verdict,
        "reasons": deduped,
        "operator_decision_id": governance.get("operator_decision_id"),
        "ecap_amendment_id": governance.get("ecap_amendment_id"),
        "ecap_amendment_version": governance.get("ecap_amendment_version"),
        "ecap_amendment_activation": activation,
        "normative_effect": effect,
        "evol_x_closure": "NOT_AUTHORIZED",
    }


def _evaluate_attestation(
    document: Mapping[str, Any],
    *,
    trust_root: Mapping[str, Any] | None,
    observed: Mapping[str, str] | None,
    expected_tool_digest: str | None,
    replay_ledger: set[str] | None,
    now: datetime,
) -> dict[str, Any]:
    schema_reason = _schema_reason(document)
    if schema_reason is not None:
        return {
            "verdict": "REJECTED",
            "reasons": [schema_reason],
            "limitations": [],
            "credential_class": document.get("credential_class") if isinstance(document, Mapping) else None,
        }
    reasons: list[str] = []
    limitations: list[str] = []
    stated = {key: document[key] for key in document if key != "signature"}
    binding_input = {key: stated[key] for key in stated if key != "replay_binding"}
    expected_binding = hashlib.sha256(canonical_bytes(binding_input)).hexdigest()
    if not _same_hex(str(stated["replay_binding"]), expected_binding):
        reasons.append("REPLAY_BINDING_MISMATCH")
    key, trust_reasons = _select_trust_key(document, trust_root)
    reasons.extend(trust_reasons)
    if key is not None and not _signature_valid(key, document):
        reasons.append("FORGED_SIGNATURE")
    reasons.extend(_semantic_reasons(document, expected_tool_digest=expected_tool_digest, now=now))
    if expected_tool_digest is None:
        limitations.append("TOOL_DIGEST_UNBOUND")
    if observed is None:
        limitations.append("MODE_RESULTS_NOT_OBSERVED")
    else:
        mode_a = document["mode_a_result"]
        mode_b = document["mode_b_result"]
        if observed.get("mode_a") != mode_a["status"] or observed.get("mode_b") != mode_b["status"]:
            reasons.append("ALTERED_MODE_RESULTS")
    if replay_ledger is None:
        limitations.append("REPLAY_PROTECTION_UNCONFIGURED")
    elif str(document["replay_binding"]) in replay_ledger:
        reasons.append("REPLAYED_ATTESTATION")
    reasons = list(dict.fromkeys(reasons))
    limitations = list(dict.fromkeys(limitations))
    credential = document["credential_class"]
    verdict = _attestation_verdict(document, reasons, limitations, trust_root)
    return {
        "verdict": verdict,
        "reasons": reasons,
        "limitations": limitations,
        "credential_class": credential,
    }


def _attestation_verdict(
    document: Mapping[str, Any],
    reasons: list[str],
    limitations: list[str],
    trust_root: Mapping[str, Any] | None,
) -> str:
    if reasons:
        return "REJECTED"
    credential = document["credential_class"]
    disposition = document["formal_acceptance_disposition"]
    custody = None if trust_root is None else trust_root.get("custody_domain")
    if credential == CREDENTIAL_TEST and disposition == "NOT_ACCEPTED" and custody == CREDENTIAL_TEST:
        return "VALID_NON_PRODUCTION_TEST_ONLY"
    production_ready = (
        credential == CREDENTIAL_PRODUCTION
        and disposition == "ACCEPTED_AUTOMATED_B6"
        and custody == CUSTODY_INDEPENDENT
        and "TOOL_DIGEST_UNBOUND" not in limitations
        and "MODE_RESULTS_NOT_OBSERVED" not in limitations
        and "REPLAY_PROTECTION_UNCONFIGURED" not in limitations
    )
    if production_ready:
        return "VALID_PRODUCTION_INDEPENDENT"
    if credential == CREDENTIAL_PRODUCTION and not reasons:
        return "SIGNATURE_VALID_NOT_FORMAL"
    return "REJECTED"


def _schema_reason(document: object) -> str | None:
    if not isinstance(document, dict) or set(document) != _ATTESTATION_KEYS:
        return "SCHEMA_INVALID"
    if document.get("schema_version") != SCHEMA_VERSION:
        return "SCHEMA_INVALID"
    if document.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        return "SCHEMA_INVALID"
    signature = document.get("signature")
    if not isinstance(signature, str) or signature.strip() == "":
        return "UNSIGNED_OR_INCOMPLETE"
    if _decode_b64(signature) is None:
        return "SCHEMA_INVALID"
    string_fields = (
        "ecap_policy_id",
        "ecap_policy_version",
        "operator_decision_ref",
        "certification_profile",
        "verification_tool_id",
        "verification_tool_version",
        "attestor_identity",
        "evidence_producer_identity",
        "key_id",
        "trust_root_ref",
        "attestation_id",
    )
    if any(not isinstance(document[field], str) or document[field].strip() == "" for field in string_fields):
        return "SCHEMA_INVALID"
    if not _is_sha(document.get("test_execution_sha"), 40):
        return "SCHEMA_INVALID"
    if not _is_sha(document.get("evidence_final_publication_sha"), 40):
        return "SCHEMA_INVALID"
    if not _is_sha(document.get("verification_tool_digest"), 64):
        return "SCHEMA_INVALID"
    if not _is_sha(document.get("replay_binding"), 64):
        return "SCHEMA_INVALID"
    if document.get("credential_class") not in {CREDENTIAL_TEST, CREDENTIAL_PRODUCTION}:
        return "SCHEMA_INVALID"
    if document.get("formal_acceptance_disposition") not in {"NOT_ACCEPTED", "ACCEPTED_AUTOMATED_B6"}:
        return "SCHEMA_INVALID"
    for field in ("signing_timestamp", "validity_not_before", "validity_not_after"):
        if _parse_time(document.get(field)) is None:
            return "SCHEMA_INVALID"
    if _nested_reason(document.get("ci_provenance"), _CI_KEYS, str) is not None:
        return "SCHEMA_INVALID"
    if _nested_reason(document.get("mode_a_result"), _MODE_KEYS, str) is not None:
        return "SCHEMA_INVALID"
    if _nested_reason(document.get("mode_b_result"), _MODE_KEYS, str) is not None:
        return "SCHEMA_INVALID"
    findings = document.get("findings_disposition")
    if not isinstance(findings, dict) or set(findings) != _FINDINGS_KEYS:
        return "SCHEMA_INVALID"
    if not isinstance(findings.get("register_id"), str) or findings["register_id"].strip() == "":
        return "SCHEMA_INVALID"
    if not isinstance(findings.get("complete"), bool):
        return "SCHEMA_INVALID"
    if not _is_int(findings.get("open_blocking")) or findings["open_blocking"] < 0:
        return "SCHEMA_INVALID"
    if not isinstance(findings.get("m09h_disposition"), str) or not isinstance(findings.get("m09a_disposition"), str):
        return "SCHEMA_INVALID"
    frz = document.get("frz_scope_declaration")
    if not isinstance(frz, dict) or set(frz) != _FRZ_KEYS:
        return "SCHEMA_INVALID"
    if not isinstance(frz.get("scope"), str) or not isinstance(frz.get("frz_evol_07"), str):
        return "SCHEMA_INVALID"
    if not isinstance(frz.get("promotes_frz"), bool) or not isinstance(frz.get("declares_evol_x_closed"), bool):
        return "SCHEMA_INVALID"
    return None


def _nested_reason(value: object, keys: frozenset[str], value_type: type[object]) -> str | None:
    if not isinstance(value, dict) or set(value) != keys:
        return "SCHEMA_INVALID"
    if any(not isinstance(item, value_type) for item in value.values()):
        return "SCHEMA_INVALID"
    return None


def _semantic_reasons(document: Mapping[str, Any], *, expected_tool_digest: str | None, now: datetime) -> list[str]:
    reasons: list[str] = []
    if document["ecap_policy_id"] != POLICY_ID:
        reasons.append("WRONG_POLICY_ID")
    if document["ecap_policy_version"] != POLICY_VERSION:
        reasons.append("WRONG_POLICY_VERSION")
    if document["operator_decision_ref"] != OPERATOR_DECISION:
        reasons.append("INSUFFICIENT_AUTHORITY")
    if document["certification_profile"] != PROFILE:
        reasons.append("WRONG_PROFILE")
    if document["test_execution_sha"] != PINNED_TES:
        reasons.append("WRONG_TES")
    if document["evidence_final_publication_sha"] != PINNED_FINAL:
        reasons.append("WRONG_FINAL")
    if document["verification_tool_id"] != TOOL_ID or document["verification_tool_version"] != TOOL_VERSION:
        reasons.append("WRONG_TOOL_IDENTITY")
    if expected_tool_digest is not None and not _same_hex(document["verification_tool_digest"], expected_tool_digest):
        reasons.append("MUTATED_ARTIFACTS")
    ci = document["ci_provenance"]
    if ci["run_id"].strip() == "" or ci["conclusion"] != "success":
        reasons.append("MISSING_CI_EVIDENCE")
    elif (
        ci["run_id"] != QUALIFIED_CI_RUN_ID
        or ci["head_sha"] != document["test_execution_sha"]
        or ci["profile"] != "evol_x_p3_l3"
    ):
        reasons.append("CI_PROVENANCE_MISMATCH")
    m09h_values = (
        document["mode_a_result"]["m09h"],
        document["mode_b_result"]["m09h"],
        document["findings_disposition"]["m09h_disposition"],
    )
    if any(value == "PASS" for value in m09h_values):
        reasons.append("M09H_PASS_CLAIM")
    elif any(value != "NOT_MEASURED" for value in m09h_values):
        reasons.append("M09H_DISPOSITION_INVALID")
    m09a_values = (
        document["mode_a_result"]["m09a"],
        document["mode_b_result"]["m09a"],
        document["findings_disposition"]["m09a_disposition"],
    )
    if any(value != "MEASURED_PASS" for value in m09a_values):
        reasons.append("M09A_DISPOSITION_INVALID")
    findings = document["findings_disposition"]
    if (
        findings["register_id"] != FINDINGS_REGISTER_ID
        or findings["complete"] is not True
        or findings["open_blocking"] != 0
    ):
        reasons.append("INCOMPLETE_FINDINGS")
    frz = document["frz_scope_declaration"]
    if frz["promotes_frz"] is True or frz["scope"] != "EVOL-X-ONLY" or frz["frz_evol_07"] == "PASS":
        reasons.append("UNSUPPORTED_FRZ_PROMOTION")
    if frz["declares_evol_x_closed"] is True:
        reasons.append("UNAUTHORIZED_CLOSURE_CLAIM")
    not_before = _parse_time(document["validity_not_before"])
    not_after = _parse_time(document["validity_not_after"])
    signed_at = _parse_time(document["signing_timestamp"])
    assert not_before is not None and not_after is not None and signed_at is not None
    if now < not_before or now > not_after or signed_at < not_before or signed_at > not_after:
        reasons.append("STALE_ATTESTATION")
    attestor = document["attestor_identity"].strip().lower()
    producer = document["evidence_producer_identity"].strip().lower()
    if attestor == producer:
        reasons.append("SELF_ACCEPTANCE")
    if attestor in IMPLEMENTER_IDENTITIES or producer in IMPLEMENTER_IDENTITIES:
        reasons.append("IMPLEMENTER_CONTROLLED_ATTESTOR")
    if document["credential_class"] == CREDENTIAL_TEST and document["formal_acceptance_disposition"] != "NOT_ACCEPTED":
        reasons.append("TEST_CREDENTIAL_FORMAL_CLAIM")
    return reasons


def _select_trust_key(
    document: Mapping[str, Any], trust_root: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if not isinstance(trust_root, dict) or set(trust_root) != _TRUST_ROOT_KEYS:
        return None, ["UNKNOWN_TRUST_ROOT"]
    if trust_root.get("schema_version") != TRUST_ROOT_SCHEMA or not isinstance(trust_root.get("trust_root_id"), str):
        return None, ["UNKNOWN_TRUST_ROOT"]
    if trust_root.get("custody_domain") not in {CREDENTIAL_TEST, CUSTODY_INDEPENDENT}:
        return None, ["UNKNOWN_TRUST_ROOT"]
    if trust_root["trust_root_id"] != document["trust_root_ref"]:
        reasons.append("UNKNOWN_TRUST_ROOT")
    keys = trust_root.get("keys")
    if not isinstance(keys, list):
        return None, ["UNKNOWN_TRUST_ROOT"]
    selected: dict[str, Any] | None = None
    for item in keys:
        if not isinstance(item, dict) or set(item) != _TRUST_KEY_KEYS:
            return None, ["UNKNOWN_TRUST_ROOT"]
        if item.get("key_id") == document["key_id"]:
            selected = item
    if selected is None:
        reasons.append("UNKNOWN_TRUST_ROOT")
        return None, list(dict.fromkeys(reasons))
    if selected.get("status") == "REVOKED":
        reasons.append("REVOKED_SIGNING_KEY")
    elif selected.get("status") != "ACTIVE":
        reasons.append("UNKNOWN_TRUST_ROOT")
    if selected.get("controlled_by_implementer") is True:
        reasons.append("IMPLEMENTER_CONTROLLED_ATTESTOR")
    if selected.get("attestor_identity") != document["attestor_identity"]:
        reasons.append("WRONG_ATTESTOR")
    if selected.get("credential_class") != document["credential_class"]:
        reasons.append("CREDENTIAL_CLASS_MISMATCH")
    custody_domain = trust_root.get("custody_domain")
    if document["credential_class"] == CREDENTIAL_PRODUCTION and custody_domain != CUSTODY_INDEPENDENT:
        reasons.append("KEY_CUSTODY_NOT_INDEPENDENT")
    if _decode_b64(selected.get("public_key_b64")) is None:
        reasons.append("UNKNOWN_TRUST_ROOT")
    return selected, list(dict.fromkeys(reasons))


def _signature_valid(key: Mapping[str, Any], document: Mapping[str, Any]) -> bool:
    public_raw = _decode_b64(key.get("public_key_b64"))
    signature = _decode_b64(document.get("signature"))
    if public_raw is None or signature is None or len(public_raw) != 32:
        return False
    signed = {field: document[field] for field in document if field != "signature"}
    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(signature, canonical_bytes(signed))
    except (InvalidSignature, ValueError):
        return False
    return True


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _is_sha(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(char in "0123456789abcdef" for char in value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same_hex(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left.encode("ascii"), right.encode("ascii"))


def _decode_b64(value: object) -> bytes | None:
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        return base64.b64decode(value, validate=True)
    except ValueError:
        return None


def public_key_b64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")
