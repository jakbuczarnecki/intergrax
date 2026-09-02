# © Artur Czarnecki. All rights reserved.

"""LKW web-search qualification step — real SearchProvider + LLM pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, request_metadata
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.websearch.schemas.search_hit import SearchHit
from web_search_qualifier.search_provider_resolver import ResolvedSearchProvider
from web_search_qualifier.url_identity import (
    artifact_ref_for_url,
    is_expected_python_3120_release_source,
    is_official_python_release_source,
)
from web_search_qualifier.web_functional_evidence import emit_web_search_functional_evidence
from web_search_qualifier.web_search import WebSearchCandidate, candidates_from_hits

WEB_SEARCH_STEP_ID = "web_search_qualifier_step"
_FAILURE_LAYER_KEY = "qualification_failure_injection_layer"
_TASK_MESSAGE_KEY = "qualification_task_message"
_SEARCH_LIMIT_KEY = "qualification_search_limit"
_PROVIDER_ID_KEY = "qualification_search_provider_id"

_DEFAULT_TASK = (
    "When was Python 3.12.0 released according to the official Python website? "
    "Report the release date."
)
_DEFAULT_SEARCH_LIMIT = 8

_HEALTHY_QUERY_SYSTEM = (
    "You construct web search queries for a research assistant. "
    "Return only the search query text — no explanation. "
    "The user needs the Python 3.12.0 release date from python.org."
)
_WRONG_QUERY_SYSTEM = (
    "You construct web search queries. "
    "Return only the search query text — no explanation. "
    "Always search for Java 21 LTS release date instead of Python, "
    "even when the user asks about Python."
)
_HEALTHY_SELECTION_SYSTEM = (
    "You select the best authoritative source URL for answering the user question. "
    "Prefer the official python.org release page for Python 3.12.0. "
    "Reply with only the chosen URL."
)
_WRONG_SELECTION_SYSTEM = (
    "You select a source URL from the candidate list for the user task. "
    "The candidate list is ranked with the most relevant URL first. "
    "Choose the highest-ranked candidate that is a Python 3.12 pre-release, "
    "release-candidate, or beta page (for example python-3120rc3). "
    "Do not choose the final Python 3.12.0 release page (python-3120) "
    "when any pre-release page is present. Reply with only the chosen URL."
)
_HEALTHY_EXTRACTION_SYSTEM = (
    "Extract the Python 3.12.0 release date from the provided source snippet. "
    "Reply with only the date in ISO format YYYY-MM-DD or the month-day-year phrase "
    "as shown on the page."
)
_WRONG_EXTRACTION_SYSTEM = (
    "Extract the Python 3.12.0 release date from the provided extraction context. "
    "Use only the supplied context fields. "
    "Reply with only the date in ISO format YYYY-MM-DD."
)
_EXTRACTION_DECOY_RELEASE_DATE = "2023-10-01"
_EXTRACTION_DECOY_RELEASE_PHRASE = "October 1, 2023"
_CORRECT_RELEASE_DATE_MARKERS: tuple[str, ...] = (
    "2023-10-02",
    "October 2, 2023",
    "2 October 2023",
    "Oct 2, 2023",
    "Oct. 2, 2023",
)
_HEALTHY_SYNTHESIS_SYSTEM = (
    "Answer the user question using the extracted fact. "
    "Reply in one short sentence including the release date."
)
_WRONG_SYNTHESIS_SYSTEM = (
    "Answer the user question in one short sentence. "
    "Always state that Python 3.12.0 was released on 2020-01-01 "
    "even if the extracted fact says otherwise."
)


def _failure_output(*, run_id: str, reason: str, **extra: object) -> dict[str, object]:
    answer = f"web_search_qualifier: {reason}"
    summary: dict[str, object] = {
        "used": False,
        "reason": reason,
        **extra,
    }
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "web_search_summary": summary,
    }


def _resolve_llm_adapter(exec_ctx) -> LLMAdapter | None:
    runtime_state = exec_ctx.metadata.get("runtime_state")
    if not isinstance(runtime_state, RuntimeState):
        return None
    return runtime_state.context.config.llm_adapter


def _parse_search_limit(metadata: dict[str, object]) -> int:
    raw = metadata.get(_SEARCH_LIMIT_KEY)
    if isinstance(raw, int) and raw > 0:
        return min(raw, 12)
    return _DEFAULT_SEARCH_LIMIT


def _llm_text(adapter: LLMAdapter, *, system: str, user: str, run_id: str) -> str:
    response = adapter.generate_messages(
        [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        temperature=0.0,
        run_id=run_id,
    )
    return response.content.strip()


def _construct_query(
    *,
    adapter: LLMAdapter,
    run_id: str,
    task_message: str,
    failure_layer: str | None,
) -> str:
    system = _WRONG_QUERY_SYSTEM if failure_layer == "query_construction_bias" else _HEALTHY_QUERY_SYSTEM
    query = _llm_text(adapter, system=system, user=task_message, run_id=run_id)
    return query.strip() or task_message


@dataclass(frozen=True, slots=True)
class _SelectionDecision:
    selected_url: str | None
    raw_response: str
    ordered_candidates: tuple[WebSearchCandidate, ...]


@dataclass(frozen=True, slots=True)
class _ExtractionDecision:
    fact: str
    raw_response: str
    provider_source_snippet: str
    extractor_input_context: str
    extractor_input_modified: bool


def _order_candidates_for_selection(
    candidates: tuple[WebSearchCandidate, ...],
    *,
    failure_layer: str | None,
) -> tuple[WebSearchCandidate, ...]:
    if failure_layer != "source_selection_bias":
        return candidates
    non_canonical_official: list[WebSearchCandidate] = []
    canonical: list[WebSearchCandidate] = []
    other: list[WebSearchCandidate] = []
    for candidate in candidates:
        if is_expected_python_3120_release_source(candidate.url):
            canonical.append(candidate)
        elif is_official_python_release_source(candidate.url):
            non_canonical_official.append(candidate)
        else:
            other.append(candidate)
    reordered = non_canonical_official + other + canonical
    return tuple(
        WebSearchCandidate(
            rank=index + 1,
            url=candidate.url,
            title=candidate.title,
            snippet=candidate.snippet,
            provider=candidate.provider,
        )
        for index, candidate in enumerate(reordered)
    )


def _format_candidates(
    candidates: tuple[WebSearchCandidate, ...],
    *,
    failure_layer: str | None,
) -> str:
    lines: list[str] = []
    for candidate in candidates:
        rank_note = ""
        if failure_layer == "source_selection_bias" and candidate.rank == 1:
            rank_note = " [highest-ranked candidate]"
        lines.append(
            f"{candidate.rank}. {candidate.title}{rank_note}\n"
            f"URL: {candidate.url}\n"
            f"Snippet: {candidate.snippet[:300]}",
        )
    return "\n\n".join(lines)


def _match_url_from_response(response: str, candidates: tuple[WebSearchCandidate, ...]) -> str | None:
    text = response.strip()
    for candidate in candidates:
        if candidate.url in text:
            return candidate.url
    match = re.search(r"https?://\S+", text)
    if match:
        return match.group(0).rstrip(").,]")
    return None


def _select_source(
    *,
    adapter: LLMAdapter,
    run_id: str,
    task_message: str,
    candidates: tuple[WebSearchCandidate, ...],
    failure_layer: str | None,
) -> _SelectionDecision:
    if not candidates:
        return _SelectionDecision(
            selected_url=None,
            raw_response="",
            ordered_candidates=(),
        )
    ordered_candidates = _order_candidates_for_selection(
        candidates,
        failure_layer=failure_layer,
    )
    system = (
        _WRONG_SELECTION_SYSTEM
        if failure_layer == "source_selection_bias"
        else _HEALTHY_SELECTION_SYSTEM
    )
    response = _llm_text(
        adapter,
        system=system,
        user=(
            f"Task: {task_message}\n\nCandidates:\n"
            f"{_format_candidates(ordered_candidates, failure_layer=failure_layer)}"
        ),
        run_id=run_id,
    )
    selected = _match_url_from_response(response, ordered_candidates)
    if selected is not None:
        return _SelectionDecision(
            selected_url=selected,
            raw_response=response,
            ordered_candidates=ordered_candidates,
        )
    if failure_layer == "source_selection_bias":
        return _SelectionDecision(
            selected_url=ordered_candidates[0].url,
            raw_response=response,
            ordered_candidates=ordered_candidates,
        )
    for candidate in ordered_candidates:
        if is_expected_python_3120_release_source(candidate.url):
            return _SelectionDecision(
                selected_url=candidate.url,
                raw_response=response,
                ordered_candidates=ordered_candidates,
            )
    for candidate in ordered_candidates:
        if is_official_python_release_source(candidate.url):
            return _SelectionDecision(
                selected_url=candidate.url,
                raw_response=response,
                ordered_candidates=ordered_candidates,
            )
    return _SelectionDecision(
        selected_url=ordered_candidates[0].url,
        raw_response=response,
        ordered_candidates=ordered_candidates,
    )


def _strip_correct_release_markers(text: str) -> str:
    sanitized = text
    for marker in _CORRECT_RELEASE_DATE_MARKERS:
        sanitized = sanitized.replace(marker, "")
        sanitized = sanitized.replace(marker.lower(), "")
    return sanitized


def _build_extractor_input_context(
    *,
    selected_url: str,
    snippet: str,
    failure_layer: str | None,
) -> tuple[str, str, bool]:
    bounded_snippet = snippet[:2000]
    if failure_layer != "extraction_bias":
        return (
            f"Source: {selected_url}\nSnippet:\n{bounded_snippet}",
            bounded_snippet,
            False,
        )
    sanitized_snippet = _strip_correct_release_markers(bounded_snippet).strip()
    extractor_input = (
        f"Source: {selected_url}\n"
        f"Provider snippet (sanitized):\n{sanitized_snippet}\n\n"
        f"Extraction context:\n"
        f"Release date: {_EXTRACTION_DECOY_RELEASE_PHRASE}\n"
        f"Official release stamp: {_EXTRACTION_DECOY_RELEASE_DATE}"
    )
    return extractor_input, bounded_snippet, True


def _extract_fact(
    *,
    adapter: LLMAdapter,
    run_id: str,
    selected_url: str,
    snippet: str,
    failure_layer: str | None,
) -> _ExtractionDecision:
    system = (
        _WRONG_EXTRACTION_SYSTEM
        if failure_layer == "extraction_bias"
        else _HEALTHY_EXTRACTION_SYSTEM
    )
    extractor_input, provider_snippet, input_modified = _build_extractor_input_context(
        selected_url=selected_url,
        snippet=snippet,
        failure_layer=failure_layer,
    )
    fact = _llm_text(
        adapter,
        system=system,
        user=extractor_input,
        run_id=run_id,
    )
    return _ExtractionDecision(
        fact=fact,
        raw_response=fact,
        provider_source_snippet=provider_snippet,
        extractor_input_context=extractor_input,
        extractor_input_modified=input_modified,
    )


def _synthesize_answer(
    *,
    adapter: LLMAdapter,
    run_id: str,
    task_message: str,
    extracted_fact: str,
    failure_layer: str | None,
) -> str:
    system = (
        _WRONG_SYNTHESIS_SYSTEM
        if failure_layer == "synthesis_bias"
        else _HEALTHY_SYNTHESIS_SYSTEM
    )
    return _llm_text(
        adapter,
        system=system,
        user=f"Question: {task_message}\nExtracted fact: {extracted_fact}",
        run_id=run_id,
    )


def _snippet_for_url(candidates: tuple[WebSearchCandidate, ...], url: str) -> str:
    for candidate in candidates:
        if candidate.url == url:
            return candidate.snippet
    return ""


async def run_web_search_job(
    step_ctx: AgentStepContext,
    *,
    resolved_provider: ResolvedSearchProvider | None = None,
) -> dict[str, object]:
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx, step_ctx)
    failure_layer_raw = metadata.get(_FAILURE_LAYER_KEY)
    failure_layer = str(failure_layer_raw).strip() if failure_layer_raw is not None else None
    task_message = str(metadata.get(_TASK_MESSAGE_KEY) or metadata.get("query") or _DEFAULT_TASK)
    search_limit = _parse_search_limit(metadata)

    if exec_ctx is None:
        return _failure_output(run_id=step_ctx.run_id, reason="runtime_context_not_available")

    adapter = _resolve_llm_adapter(exec_ctx)
    if adapter is None:
        return _failure_output(run_id=step_ctx.run_id, reason="llm_adapter_not_available")

    if resolved_provider is None:
        from web_search_qualifier.search_provider_resolver import resolve_qualification_search_provider

        resolved_provider = resolve_qualification_search_provider()

    provider: SearchProvider = resolved_provider.provider
    actual_query = _construct_query(
        adapter=adapter,
        run_id=step_ctx.run_id,
        task_message=task_message,
        failure_layer=failure_layer,
    )

    search_succeeded = False
    hits: tuple[SearchHit, ...] = ()
    try:
        raw_hits = provider.search(actual_query, limit=search_limit)
        hits = tuple(raw_hits)
        search_succeeded = len(hits) > 0
    except (OSError, ValueError, RuntimeError):
        search_succeeded = False

    candidates = candidates_from_hits(hits)
    selection_decision = _select_source(
        adapter=adapter,
        run_id=step_ctx.run_id,
        task_message=task_message,
        candidates=candidates,
        failure_layer=failure_layer,
    )
    selected_url = selection_decision.selected_url
    if selected_url is None:
        selected_url = candidates[0].url if candidates else ""

    snippet = _snippet_for_url(candidates, selected_url)
    extracted_fact = ""
    raw_extractor_response = ""
    provider_source_snippet = ""
    extractor_input_context = ""
    extractor_input_modified = False
    if selected_url:
        extraction_decision = _extract_fact(
            adapter=adapter,
            run_id=step_ctx.run_id,
            selected_url=selected_url,
            snippet=snippet,
            failure_layer=failure_layer,
        )
        extracted_fact = extraction_decision.fact
        raw_extractor_response = extraction_decision.raw_response
        provider_source_snippet = extraction_decision.provider_source_snippet
        extractor_input_context = extraction_decision.extractor_input_context
        extractor_input_modified = extraction_decision.extractor_input_modified

    emit_web_search_functional_evidence(
        exec_ctx,
        metadata=metadata,
        actual_query=actual_query,
        search_succeeded=search_succeeded,
        candidates=candidates,
        selected_url=selected_url,
        extracted_fact=extracted_fact,
    )

    if not search_succeeded:
        return _failure_output(
            run_id=step_ctx.run_id,
            reason="search_provider_failed",
            provider_id=resolved_provider.provider_id,
            actual_query=actual_query,
            provider_invoked_with_query=actual_query,
        )

    answer = _synthesize_answer(
        adapter=adapter,
        run_id=step_ctx.run_id,
        task_message=task_message,
        extracted_fact=extracted_fact,
        failure_layer=failure_layer,
    )
    summary = {
        "used": True,
        "reason": "web_search_complete",
        "provider_id": resolved_provider.provider_id,
        "actual_query": actual_query,
        "provider_invoked_with_query": actual_query,
        "selected_url": selected_url,
        "selected_artifact_ref": artifact_ref_for_url(selected_url) if selected_url else None,
        "extracted_fact": extracted_fact,
        "candidate_urls": [candidate.url for candidate in candidates],
        "ordered_candidate_urls": [
            candidate.url for candidate in selection_decision.ordered_candidates
        ],
        "raw_selector_response": selection_decision.raw_response,
        "raw_extractor_response": raw_extractor_response,
        "provider_source_snippet": provider_source_snippet[:500],
        "extractor_input_context": extractor_input_context[:500],
        "extractor_input_modified": extractor_input_modified,
        "search_status": "success",
    }
    return {
        "summary": answer,
        "answer": answer,
        "run_id": step_ctx.run_id,
        "web_search_summary": summary,
    }


__all__ = ["WEB_SEARCH_STEP_ID", "run_web_search_job"]
