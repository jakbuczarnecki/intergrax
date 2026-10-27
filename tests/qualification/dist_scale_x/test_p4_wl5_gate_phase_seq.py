# © Artur Czarnecki. All rights reserved.

"""Regression: WL-5 gate seq must not collide across phases."""

from __future__ import annotations

from pathlib import Path

from tests.qualification.dist_scale_x.p4_wl5_gate import Wl5RateGate


def test_wl5_gate_second_phase_offers_unique_and_claimable(tmp_path: Path) -> None:
    gate = Wl5RateGate(tmp_path / "gate.db")
    warmup_count = 6
    for _ in range(warmup_count):
        gate.offer_next()
    for _ in range(warmup_count):
        assert gate.try_claim("w0") is not None
    assert gate.pending_offers() == 0

    measure_count = 4
    measure_seqs = [gate.offer_next() for _ in range(measure_count)]
    assert measure_seqs == list(range(warmup_count, warmup_count + measure_count))
    assert gate.pending_offers() == measure_count

    claimed = [gate.try_claim("w1") for _ in range(measure_count)]
    assert claimed == measure_seqs
    assert gate.pending_offers() == 0

    # INSERT OR IGNORE on reused seq must not resurrect dropped measure offers
    gate.offer(0)
    assert gate.pending_offers() == 0
