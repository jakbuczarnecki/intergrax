# © Artur Czarnecki. All rights reserved.

"""Deterministic stub domain logic for K.1 prototype (no external network)."""

from __future__ import annotations

from problem_radar.schemas.output import ProblemCluster, ProblemRadarOutput


def build_stub_problem_radar_output(query: str) -> ProblemRadarOutput:
    """
    Produce a typed stub report from the user query.

    Live source ingestion (HN, Reddit, …) replaces this in later K.1 waves.
    """
    topic = (query or "unspecified market").strip()[:120]
    cluster = ProblemCluster(
        cluster_id="stub-1",
        title=f"Recurring pain around: {topic}",
        representative_quotes=[
            f"[stub] Users report friction when working with {topic}.",
            f"[stub] Repeated complaints about slow or opaque workflows for {topic}.",
        ],
        source_links=["stub://example/hn-thread", "stub://example/reddit-thread"],
        frequency_estimate=0.42,
        intensity_score=0.61,
        affected_user_group="early adopters / builders",
        possible_product_ideas=[
            f"Vertical workflow tool for {topic}",
            f"Monitoring dashboard for {topic} complaints",
        ],
        mom_test_risk_notes=["Stub data — validate with real interviews before building."],
        confidence=0.35,
    )
    return ProblemRadarOutput(
        clusters=[cluster],
        summary=(
            f"Prototype scan for '{topic}': one stub cluster. "
            "Wire websearch + ingestion in the next K.1 iteration."
        ),
        confidence=0.35,
    )
