# © Artur Czarnecki. All rights reserved.

"""Deterministic stub domain logic for K.2 prototype (no external network)."""

from __future__ import annotations

from vendor_discovery.schemas.output import VendorCandidate, VendorDiscoveryOutput


def build_stub_vendor_discovery_output(query: str) -> VendorDiscoveryOutput:
    need = (query or "unspecified procurement need").strip()[:120]
    candidate = VendorCandidate(
        vendor_id="stub-1",
        name=f"Example vendor for {need}",
        category="saas",
        fit_score=0.48,
        risk_notes=["Stub data — validate with real vendor diligence."],
        source_links=["stub://example/vendor-profile"],
    )
    return VendorDiscoveryOutput(
        candidates=[candidate],
        summary=(
            f"Prototype vendor scan for '{need}': one stub candidate. "
            "Wire integrations + scoring in the next K.2 iteration."
        ),
        confidence=0.32,
    )
