# © Artur Czarnecki. All rights reserved.

"""Per-concern architecture disposition for RPL-H final convergence (Phase A/C SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_b6_open_concern_ids, normative_rpl_row


class ActualContractKind(StrEnum):
    REPLACEABLE_INTEGRATION = "replaceable_integration"
    REPLACEABLE_PLATFORM = "replaceable_platform"
    INTERNAL_COMPOSITION = "internal_composition"
    FIXED_VENDOR_SINGLE = "fixed_vendor_single"
    ARCHITECTURE_GAP = "architecture_gap"


@dataclass(frozen=True, slots=True)
class RplConvergenceDisposition:
    concern_id: str
    actual_contract: ActualContractKind
    production_owner: str
    selection_mechanism: str
    current_evidence: str
    missing_requirement: str
    recommendation: str
    options: str
    architecture_justification: str
    decision_required: bool
    b6_impact: str


_FACTORY_DISPOSITION: Final[dict[str, RplConvergenceDisposition]] = {
    "agent.effective_profile": RplConvergenceDisposition(
        concern_id="agent.effective_profile",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="agents/ — effective profile resolution",
        selection_mechanism="Profile/bootstrap composition (not integration catalog dual-slug)",
        current_evidence="Package C factory ownership read — same node pos/neg",
        missing_requirement="Distinct alternate effective-profile backend or INTERNAL reclassification @ ADR",
        recommendation="Add behavioral dual-path proof or ratify INTERNAL (not claimed replaceable)",
        options="Dual profile fixture; OD-RPL-G2 ALLOWED_FIXED; demote matrix extensibility",
        architecture_justification="Matrix claims REPLACEABLE NOT YET; proof is structural factory reachability only",
        decision_required=True,
        b6_impact="Blocks B6 until QUALIFIED or scoped OD",
    ),
    "application.composition_root": RplConvergenceDisposition(
        concern_id="application.composition_root",
        actual_contract=ActualContractKind.INTERNAL_COMPOSITION,
        production_owner="applications/ composition roots",
        selection_mechanism="Per-application composition module — not vendor slug",
        current_evidence="Package C factory.py ownership",
        missing_requirement="Replaceability proof inappropriate without per-app second root; reclassify or scoped OD",
        recommendation="Operator: confirm INTERNAL composition vs require two sanctioned roots",
        options="Document INTERNAL; second reference app root; defer B6",
        architecture_justification="Tier-3 composition is app-scoped; matrix row over-promises vendor replaceability",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "distributed.sqlite_kv": RplConvergenceDisposition(
        concern_id="distributed.sqlite_kv",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/distributed/providers/sqlite_kv_store.py",
        selection_mechanism="Distributed provider registration",
        current_evidence="Package C factory ownership",
        missing_requirement="Second distributed KV backend behavioral equivalence",
        recommendation="Implement in-process alternate provider qualification test",
        options="Add mem/dict provider test; OD fixed substrate for sqlite lab",
        architecture_justification=(
            "SQLite lab substrate documented; alternate provider exists in architecture inventory"
        ),
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "env.integration_slug": RplConvergenceDisposition(
        concern_id="env.integration_slug",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/integrations/_shared/config.py",
        selection_mechanism="INTERGRAX_INTEGRATION_* env → read_integration_slug_from_env",
        current_evidence="Package C maps to factory ownership — env path not bound as distinct proof",
        missing_requirement="Dedicated positive env slug + negative unset proof nodes in normative SSOT",
        recommendation="Wire env slug behavioral test as normative pos/neg (authorized technical)",
        options="Extend Package C helper; OD if env override deemed non-replaceable",
        architecture_justification="Public env contract is documented selection surface",
        decision_required=True,
        b6_impact="Blocks B6 until proof wired",
    ),
    "execution.integration_bound_resolution": RplConvergenceDisposition(
        concern_id="execution.integration_bound_resolution",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/runtime execution integration bounds",
        selection_mechanism="Bound resolution at execution harness",
        current_evidence="Factory ownership only",
        missing_requirement="Alternate bound/backend resolution adversarial test",
        recommendation="Add execution-bound dual fixture or INTERNAL classification",
        options="Harness qualification test; OD scope",
        architecture_justification="Claimed in replaceability matrix without alternate demonstration",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "llm.chat_completion": RplConvergenceDisposition(
        concern_id="llm.chat_completion",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/llm adapter registry",
        selection_mechanism="LLM adapter registration + routing",
        current_evidence="Tokenizer alternate mis-mapped in Package C concern_proof",
        missing_requirement="Two sanctioned in-process LLM adapters with routing proof",
        recommendation="Replace proxy tokenizer test with llm registry dual adapter test",
        options="Stub adapters in qualification; OD fixed single vendor",
        architecture_justification="Extensibility YES in inventory — tokenizer proxy is not LLM proof",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "llm.embedding": RplConvergenceDisposition(
        concern_id="llm.embedding",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/llm embedding adapters",
        selection_mechanism="Adapter registry",
        current_evidence="Tokenizer proxy @ Package C",
        missing_requirement="Dual embedding adapter behavioral proof",
        recommendation="Same as llm.chat_completion — real adapter pair",
        options="Qualification stubs; OD vendor Path B",
        architecture_justification="Matrix promises replaceability; proof binding incorrect",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "llm.rerank": RplConvergenceDisposition(
        concern_id="llm.rerank",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/llm rerank adapters",
        selection_mechanism="Adapter registry",
        current_evidence="Tokenizer proxy",
        missing_requirement="Dual rerank adapter proof",
        recommendation="Implement adapter pair qualification",
        options="Stubs; OD",
        architecture_justification="Incorrect proof binding",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "llm.fallback_policy": RplConvergenceDisposition(
        concern_id="llm.fallback_policy",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/llm fallback policy",
        selection_mechanism="Policy + adapter chain",
        current_evidence="Tokenizer proxy",
        missing_requirement="Fallback chain with two adapters demonstrated",
        recommendation="Behavioral fallback qualification",
        options="Stubs; OD",
        architecture_justification="Policy replaceability requires chain proof not tokenizer",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "llm.model_routing_qualifier": RplConvergenceDisposition(
        concern_id="llm.model_routing_qualifier",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/llm routing",
        selection_mechanism="Qualifier registry",
        current_evidence="Tokenizer proxy",
        missing_requirement="Dual qualifier / route proof",
        recommendation="Dedicated routing qualification",
        options="Stubs; OD",
        architecture_justification="Mis-bound Package C proof",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "plugin.integration_catalog": RplConvergenceDisposition(
        concern_id="plugin.integration_catalog",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/integrations/registry/catalog.py",
        selection_mechanism="register_integration_plugin + catalog",
        current_evidence="Factory ownership",
        missing_requirement="Second plugin registration path with selection proof",
        recommendation="Catalog dual-plugin qualification test",
        options="In-process plugins; OD",
        architecture_justification="Registry extensibility claimed",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "plugin.llm_adapter_registry": RplConvergenceDisposition(
        concern_id="plugin.llm_adapter_registry",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/llm registry",
        selection_mechanism="LLM adapter plugin registration",
        current_evidence="Factory ownership",
        missing_requirement="Two adapters registered + selected",
        recommendation="Mirror integration catalog dual registration test",
        options="Stubs; OD",
        architecture_justification="Replaceability claimed at registry layer",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "tool.marketplace_activation": RplConvergenceDisposition(
        concern_id="tool.marketplace_activation",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="agents/ tool marketplace",
        selection_mechanism="Marketplace activation gate",
        current_evidence="Factory ownership",
        missing_requirement="Alternate activation backend or INTERNAL",
        recommendation="Operator classification + proof or OD",
        options="Second marketplace stub; scope residual",
        architecture_justification="Tier-2 tool surface — replaceability not demonstrated",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "speech.provider": RplConvergenceDisposition(
        concern_id="speech.provider",
        actual_contract=ActualContractKind.REPLACEABLE_INTEGRATION,
        production_owner="intergrax/integrations speech_provider category",
        selection_mechanism="Integration catalog (speech in dual-slug catalog separately)",
        current_evidence="Factory ownership — speech_provider also in dual-slug catalog as qualified",
        missing_requirement="Factory row redundant with integration.speech_provider QUALIFIED via CDE",
        recommendation="Reconcile matrix duplicate concern speech.provider vs integration.speech_provider",
        options="Demote factory row; link to integration concern",
        architecture_justification="Potential duplicate concern ID in matrix",
        decision_required=True,
        b6_impact="Blocks B6 for factory row until reconciled",
    ),
    "tool.observability_backend": RplConvergenceDisposition(
        concern_id="tool.observability_backend",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="agents/ tool observability hook",
        selection_mechanism="Tool runtime observability binding",
        current_evidence="Factory ownership",
        missing_requirement="Distinct observability backend selection proof",
        recommendation="Qualification test or INTERNAL",
        options="Dual backend stub; OD",
        architecture_justification="Not integration catalog — platform tool hook",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "memory.sqlite.conversational": RplConvergenceDisposition(
        concern_id="memory.sqlite.conversational",
        actual_contract=ActualContractKind.REPLACEABLE_PLATFORM,
        production_owner="intergrax/memory sqlite conversational",
        selection_mechanism="Memory provider selection",
        current_evidence="Factory ownership",
        missing_requirement="Alternate memory backend proof",
        recommendation="Second in-process memory provider test or fixed substrate OD",
        options="In-memory alternate; sqlite lab fixed",
        architecture_justification="SQLite conversational lab — replaceability partially promised",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
    "prodq.colocated_runtime_adapter_isinstance": RplConvergenceDisposition(
        concern_id="prodq.colocated_runtime_adapter_isinstance",
        actual_contract=ActualContractKind.INTERNAL_COMPOSITION,
        production_owner="applications/ prod-q colocated adapter",
        selection_mechanism="Runtime adapter isinstance gate for qualification",
        current_evidence="Factory ownership",
        missing_requirement="Clarify INTERNAL qualification gate vs replaceability",
        recommendation="Operator ratify INTERNAL IMPLEMENTATION (PROD-Q gate only)",
        options="OD scope; add second adapter type for true replaceability",
        architecture_justification="PROD-Q isinstance is acceptance gate not vendor neutrality",
        decision_required=True,
        b6_impact="Blocks B6",
    ),
}

_VENDOR_BLOCKED: Final[dict[str, RplConvergenceDisposition]] = {
    "integration.external_work": RplConvergenceDisposition(
        concern_id="integration.external_work",
        actual_contract=ActualContractKind.ARCHITECTURE_GAP,
        production_owner="intergrax/contracts/external_work + governed contractor adapter",
        selection_mechanism="External work provider capabilities — not IntegrationCategory catalog",
        current_evidence="Package C factory fail-closed only",
        missing_requirement="Second external-work provider registration without new category OR OD fixed substrate",
        recommendation="Architecture: sanctioned second in-process external_work provider OR OD-RPL-G2",
        options="Add qualification provider module; defer live vendor; ALLOWED_FIXED ADR",
        architecture_justification="No providers/external_work — dual-slug catalog inapplicable; not mock-registry fix",
        decision_required=True,
        b6_impact="Blocks B6 — vendor/architecture",
    ),
    "integration.billing_meter": RplConvergenceDisposition(
        concern_id="integration.billing_meter",
        actual_contract=ActualContractKind.FIXED_VENDOR_SINGLE,
        production_owner="intergrax/integrations/providers/billing_meter/stripe",
        selection_mechanism="Single catalog slug (stripe)",
        current_evidence="Package C unregistered fail-closed",
        missing_requirement="Second billing provider @ vendor sandbox Path B",
        recommendation="OD-RPL-G2 fixed substrate OR fund second provider qualification",
        options="ALLOWED_FIXED; alternate sandbox; defer B6",
        architecture_justification="VBN single-vendor catalog — live Path B boundary",
        decision_required=True,
        b6_impact="Blocks B6 — vendor",
    ),
    "integration.managed_retrieval": RplConvergenceDisposition(
        concern_id="integration.managed_retrieval",
        actual_contract=ActualContractKind.FIXED_VENDOR_SINGLE,
        production_owner="openai managed retrieval provider",
        selection_mechanism="Catalog single slug",
        current_evidence="Package C fail-closed",
        missing_requirement="Second retrieval vendor sandbox",
        recommendation="OD-RPL-G2",
        options="Second provider; ALLOWED_FIXED",
        architecture_justification="Single catalog provider",
        decision_required=True,
        b6_impact="Blocks B6 — vendor",
    ),
    "integration.ml_inference_host": RplConvergenceDisposition(
        concern_id="integration.ml_inference_host",
        actual_contract=ActualContractKind.FIXED_VENDOR_SINGLE,
        production_owner="replicate provider",
        selection_mechanism="Catalog single slug",
        current_evidence="Package C fail-closed",
        missing_requirement="Second inference host sandbox",
        recommendation="OD-RPL-G2",
        options="Second provider; ALLOWED_FIXED",
        architecture_justification="Single catalog provider",
        decision_required=True,
        b6_impact="Blocks B6 — vendor",
    ),
    "integration.model_serving_runtime": RplConvergenceDisposition(
        concern_id="integration.model_serving_runtime",
        actual_contract=ActualContractKind.FIXED_VENDOR_SINGLE,
        production_owner="ollama runtime provider",
        selection_mechanism="Catalog single slug",
        current_evidence="Package C fail-closed",
        missing_requirement="Second serving runtime sandbox",
        recommendation="OD-RPL-G2",
        options="Second provider; ALLOWED_FIXED",
        architecture_justification="Single catalog provider",
        decision_required=True,
        b6_impact="Blocks B6 — vendor",
    ),
    "integration.vision_serving": RplConvergenceDisposition(
        concern_id="integration.vision_serving",
        actual_contract=ActualContractKind.FIXED_VENDOR_SINGLE,
        production_owner="triton vision provider",
        selection_mechanism="Catalog single slug",
        current_evidence="Package C fail-closed",
        missing_requirement="Second vision serving sandbox",
        recommendation="OD-RPL-G2",
        options="Second provider; ALLOWED_FIXED",
        architecture_justification="Single catalog provider",
        decision_required=True,
        b6_impact="Blocks B6 — vendor",
    ),
}


def open_convergence_dispositions() -> tuple[RplConvergenceDisposition, ...]:
    rows: list[RplConvergenceDisposition] = []
    for cid in normative_rpl_b6_open_concern_ids():
        if cid in _FACTORY_DISPOSITION:
            rows.append(_FACTORY_DISPOSITION[cid])
        elif cid in _VENDOR_BLOCKED:
            rows.append(_VENDOR_BLOCKED[cid])
        else:
            norm = normative_rpl_row(cid)
            rows.append(
                RplConvergenceDisposition(
                    concern_id=cid,
                    actual_contract=ActualContractKind.REPLACEABLE_INTEGRATION,
                    production_owner="intergrax/integrations",
                    selection_mechanism="See normative SSOT",
                    current_evidence=norm.positive_test,
                    missing_requirement=norm.defect,
                    recommendation="See EBH_7_B6_BLOCKER_REGISTER.md",
                    options="Qualify; OD",
                    architecture_justification=norm.acceptance,
                    decision_required=True,
                    b6_impact="Blocks B6",
                )
            )
    return tuple(rows)


def assert_ebh7_rpl_convergence_disposition_complete() -> None:
    open_ids = normative_rpl_b6_open_concern_ids()
    dispositions = {d.concern_id for d in open_convergence_dispositions()}
    assert dispositions == set(open_ids), f"Missing disposition: {set(open_ids) - dispositions}"
