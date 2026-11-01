# © Artur Czarnecki. All rights reserved.

"""EXP-S8: new application = extension only; core edit refused (frozen_core oracle; M-10)."""

from __future__ import annotations

import subprocess

from testing_support.change_safety.assess.extension_safety import ExtensionVerdict, assess_extension_safety
from tests.qualification.evol_x.experiments._harness import ArmResult, load_oracle

SCENARIO = "S8"


def tracked_scaffold_paths() -> frozenset[str]:
    """Real scaffold evidence: tracked files of the platform's POC template application."""
    out = subprocess.run(
        ["git", "ls-files", "applications/poc_template_application"], check=True, capture_output=True, text=True
    )
    return frozenset(line.strip() for line in out.stdout.splitlines() if line.strip())


def run() -> tuple[ArmResult, ArmResult]:
    oracle = load_oracle("s8_new_application")
    scaffold = assess_extension_safety(changed_paths=frozenset(oracle["scaffold_changes"]))
    real = assess_extension_safety(changed_paths=tracked_scaffold_paths())
    pos_ok = (
        scaffold.verdict is ExtensionVerdict.EXTENSION_ONLY
        and scaffold.effective_class == "A"
        and real.verdict is ExtensionVerdict.EXTENSION_ONLY
        and bool(tracked_scaffold_paths())
    )
    core = assess_extension_safety(changed_paths=frozenset(oracle["core_edit_changes"]))
    unauthorized = len(core.unauthorized_surfaces)
    neg_ok = (
        core.verdict is ExtensionVerdict.CORE_TOUCH_UNAUTHORIZED
        and list(core.unauthorized_surfaces) == [oracle["expected_core_surface"]]
        and core.effective_class == "C"
    )
    return (
        ArmResult(
            SCENARIO,
            "positive",
            pos_ok,
            f"scaffold={scaffold.verdict.value} real_poc_files={len(tracked_scaffold_paths())}",
        ),
        ArmResult(SCENARIO, "negative", neg_ok, f"core_edit={core.verdict.value} unauthorized_surfaces={unauthorized}"),
    )
