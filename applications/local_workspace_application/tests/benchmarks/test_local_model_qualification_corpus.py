# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from local_workspace_application.benchmarks.local_model_qualification.contracts import CORPUS_VERSION
from local_workspace_application.benchmarks.local_model_qualification.corpus import (
    corpus_version,
    qualification_cases,
)


def test_corpus_version_stable() -> None:
    assert corpus_version() == CORPUS_VERSION
    assert corpus_version() == "lkw.local_model_qualification.corpus.v1"


def test_twelve_unique_case_ids() -> None:
    cases = qualification_cases()
    assert len(cases) == 12
    ids = [case.case_id for case in cases]
    assert len(ids) == len(set(ids))


def test_all_requests_validate() -> None:
    for case in qualification_cases():
        dumped = case.request.model_dump()
        assert dumped


def test_all_expected_action_types_are_canonical() -> None:
    allowed = {
        "workspace.list",
        "workspace.create",
        "workspace.activate",
        "workspace.delete",
        "workspace.ask",
        "source.list",
        "source_candidate.list",
        "source_candidate.attach",
        "knowledge.add_attachments",
        "knowledge.add_sources",
    }
    for case in qualification_cases():
        assert set(case.expected.action_type_counts).issubset(allowed)


def test_first_three_critical_scenarios_exist() -> None:
    ids = {case.case_id for case in qualification_cases()}
    assert "planner.mixed_source_ordinal_routing" in ids
    assert "planner.target_workspace_without_activation" in ids
    assert "planner.explicit_workspace_activation" in ids


def test_target_without_activation_forbids_workspace_activate() -> None:
    case = next(
        case
        for case in qualification_cases()
        if case.case_id == "planner.target_workspace_without_activation"
    )
    assert "workspace.activate" in case.expected.forbidden_action_types


def test_all_inputs_are_deterministic() -> None:
    first = [case.request.model_dump_json() for case in qualification_cases()]
    second = [case.request.model_dump_json() for case in qualification_cases()]
    assert first == second
