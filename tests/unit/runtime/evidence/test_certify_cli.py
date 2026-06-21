# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.certify import run_certify_core

pytestmark = pytest.mark.unit


def test_run_certify_core_writes_report_and_returns_zero(tmp_path: Path) -> None:
    args = argparse.Namespace(level="L1", output_dir=tmp_path, root=tmp_path)
    code = run_certify_core(args)
    assert code == 0
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_run_certify_core_invalid_level_returns_nonzero(tmp_path: Path) -> None:
    args = argparse.Namespace(level="L9", output_dir=tmp_path, root=tmp_path)
    with pytest.raises(ValueError, match="invalid core certification level"):
        run_certify_core(args)
