# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.assess.extension_safety import (
    ExtensionVerdict,
    assess_extension_safety,
    assess_frozen_core_touch,
    classify_frozen_core_touches,
)

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_new_application_scaffold_is_extension_only() -> None:
    rep = assess_extension_safety(
        changed_paths=frozenset(
            {"applications/new_app/main.py", "applications/new_app/config.json", "tests/unit/x/test_new_app.py"}
        )
    )
    assert rep.verdict is ExtensionVerdict.EXTENSION_ONLY
    assert rep.effective_class == "A" and not rep.touches


def test_unauthorized_core_edit_is_flagged_class_c() -> None:
    rep = assess_extension_safety(changed_paths=frozenset({"intergrax/runtime/execution/engine.py"}))
    assert rep.verdict is ExtensionVerdict.CORE_TOUCH_UNAUTHORIZED
    assert rep.unauthorized_surfaces == ("execution",)
    assert rep.effective_class == "C"


def test_authorization_record_downgrades_to_authorized_but_keeps_class() -> None:
    rep = assess_extension_safety(
        changed_paths=frozenset({"intergrax/runtime/execution/engine.py"}),
        authorized_surfaces=frozenset({"execution"}),
    )
    assert rep.verdict is ExtensionVerdict.CORE_TOUCH_AUTHORIZED
    assert rep.effective_class == "C"  # authorization never lowers the governance class


def test_unclassified_path_is_failsafe_class_c() -> None:
    rep = assess_extension_safety(changed_paths=frozenset({"intergrax/rag/loader.py"}))
    assert rep.verdict is ExtensionVerdict.UNCLASSIFIED
    assert rep.unclassified_paths == ("intergrax/rag/loader.py",)
    assert rep.effective_class == "C"


def test_most_specific_surface_wins_and_windows_paths_normalised() -> None:
    touches = classify_frozen_core_touches(frozenset({"intergrax\\contracts\\migrations\\registry.py"}))
    assert [t.surface_id for t in touches] == ["intergrax/contracts/migrations/registry.py"]
    assert assess_frozen_core_touch(changed_paths=frozenset({"intergrax/contracts/x.py"}))
    assert not assess_frozen_core_touch(changed_paths=frozenset({"applications/a/x.py"}))
