"""Deterministic identifier and structured constraint extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingDistinguishingRequirement,
    MissingRequirementOrigin,
    NegativeAttributeConstraint,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    ExtractionCertainty,
    ExtractedConstraintRecord,
    ExtractedIdentifierRecord,
    ExtractedNegativeConstraintRecord,
    ExtractedSoftPreferenceRecord,
    QuerySourceSpan,
    QueryUnderstandingIssue,
    QueryUnderstandingIssueCode,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.normalization import (
    CANONICAL_CAPACITY,
    CANONICAL_ECC,
    CANONICAL_INTERFACE,
    CANONICAL_MEMORY_TYPE,
    _CAPACITY_PATTERN,
    _MEMORY_TYPE_PATTERN,
    normalize_capacity_token,
    normalize_ecc_required,
    normalize_identifier_value,
    normalize_interface_token,
    normalize_memory_type_token,
)

_LABELLED_IDENTIFIER = re.compile(
    r"(?P<label>GTIN|MPN|SKU|PRODUCT[\s_-]?ID)\s*[:#]?\s*"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-\._/\s]{0,96})",
    re.IGNORECASE,
)
_STRUCTURAL_GTIN = re.compile(r"(?<![A-Za-z0-9])(?P<value>\d{8}|\d{12}|\d{13}|\d{14})(?![0-9])")
_NEGATIVE_INTERFACE = re.compile(
    r"\b(?:not|without|no)\s+(?P<token>NVMe|SATA|PCIe|USB-C)\b",
    re.IGNORECASE,
)
_NEGATIVE_MEMORY = re.compile(
    r"\b(?:not|without|no|except)\s+(?P<token>DDR[345])\b",
    re.IGNORECASE,
)
_NEGATIVE_CAPACITY = re.compile(
    r"\b(?:not|without|no|except)\s+(?P<cap>\d+(?:\.\d+)?\s*(?:TB|GB|MB))\b",
    re.IGNORECASE,
)
_SOFT_CAPACITY = re.compile(
    r"\b(?:prefer|ideally|preferably)\s+(?P<cap>\d+(?:\.\d+)?\s*(?:TB|GB|MB))\b",
    re.IGNORECASE,
)
_MISSING_DISTINGUISHING = re.compile(
    r"\b(?:don'?t|do not)\s+know\s+whether\s+it\s+is\s+"
    r"(?P<a>sata|nvme|ddr[345])\b",
    re.IGNORECASE,
)


def _span(text: str, start: int, end: int) -> QuerySourceSpan:
    fragment = text[start:end] if end - start <= 64 else text[start : start + 64]
    return QuerySourceSpan(start_offset=start, end_offset=end, fragment=fragment)


def _label_to_type(label: str) -> ProductIdentifierType:
    normalized = label.upper().replace(" ", "").replace("-", "").replace("_", "")
    if normalized == "GTIN":
        return ProductIdentifierType.GTIN
    if normalized == "MPN":
        return ProductIdentifierType.MPN
    if normalized == "SKU":
        return ProductIdentifierType.SKU
    return ProductIdentifierType.PRODUCT_ID


class ProductIdentifierExtractor(Protocol):
    def extract(
        self,
        raw_text: str,
    ) -> tuple[tuple[ExtractedIdentifierRecord, ...], tuple[QueryUnderstandingIssue, ...]]:
        ...


class StructuredConstraintExtractor(Protocol):
    def extract(
        self,
        raw_text: str,
    ) -> tuple[
        tuple[ExtractedConstraintRecord, ...],
        tuple[ExtractedNegativeConstraintRecord, ...],
        tuple[ExtractedSoftPreferenceRecord, ...],
        tuple[MissingDistinguishingRequirement, ...],
        tuple[QueryUnderstandingIssue, ...],
    ]:
        ...


@dataclass(frozen=True, slots=True)
class DeterministicProductIdentifierExtractor:
    allow_structural_gtin: bool = True

    def extract(
        self,
        raw_text: str,
    ) -> tuple[tuple[ExtractedIdentifierRecord, ...], tuple[QueryUnderstandingIssue, ...]]:
        issues: list[QueryUnderstandingIssue] = []
        records: list[ExtractedIdentifierRecord] = []
        occupied: list[tuple[int, int]] = []

        def overlaps(start: int, end: int) -> bool:
            for o_start, o_end in occupied:
                if start < o_end and end > o_start:
                    return True
            return False

        for match in _LABELLED_IDENTIFIER.finditer(raw_text):
            start, end = match.start(), match.end()
            if overlaps(start, end):
                issues.append(
                    QueryUnderstandingIssue(
                        code=QueryUnderstandingIssueCode.AMBIGUOUS_IDENTIFIER_TYPE,
                        detail="overlapping labelled identifier spans",
                    )
                )
                continue
            id_type = _label_to_type(match.group("label"))
            raw_value = match.group("value").strip()
            normalized, rule = normalize_identifier_value(id_type, raw_value)
            if not normalized:
                issues.append(
                    QueryUnderstandingIssue(
                        code=QueryUnderstandingIssueCode.INVALID_IDENTIFIER,
                        detail=f"{id_type.value}:{raw_value}",
                    )
                )
                continue
            occupied.append((start, end))
            records.append(
                ExtractedIdentifierRecord(
                    identifier=ProductIdentifier(
                        identifier_type=id_type,
                        value=normalized,
                        source_field="user_request",
                    ),
                    raw_value=raw_value,
                    normalized_value=normalized,
                    normalization_rule=rule,
                    source_span=_span(raw_text, start, end),
                    certainty=ExtractionCertainty.DIRECT,
                )
            )

        if self.allow_structural_gtin:
            for match in _STRUCTURAL_GTIN.finditer(raw_text):
                start, end = match.start("value"), match.end("value")
                if overlaps(start, end):
                    continue
                raw_value = match.group("value")
                normalized, rule = normalize_identifier_value(
                    ProductIdentifierType.GTIN,
                    raw_value,
                )
                if not normalized:
                    continue
                occupied.append((start, end))
                records.append(
                    ExtractedIdentifierRecord(
                        identifier=ProductIdentifier(
                            identifier_type=ProductIdentifierType.GTIN,
                            value=normalized,
                            source_field="user_request",
                        ),
                        raw_value=raw_value,
                        normalized_value=normalized,
                        normalization_rule=rule,
                        source_span=_span(raw_text, start, end),
                        certainty=ExtractionCertainty.DETERMINISTIC,
                    )
                )

        deduped = _dedupe_identifiers(records)
        return tuple(deduped), tuple(issues)


def _dedupe_identifiers(
    records: list[ExtractedIdentifierRecord],
) -> list[ExtractedIdentifierRecord]:
    seen: set[tuple[ProductIdentifierType, str]] = set()
    ordered: list[ExtractedIdentifierRecord] = []
    for item in sorted(records, key=lambda r: (r.source_span.start_offset, r.identifier.value)):
        key = (item.identifier.identifier_type, item.identifier.value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


@dataclass(frozen=True, slots=True)
class DeterministicStructuredConstraintExtractor:
    def extract(
        self,
        raw_text: str,
    ) -> tuple[
        tuple[ExtractedConstraintRecord, ...],
        tuple[ExtractedNegativeConstraintRecord, ...],
        tuple[ExtractedSoftPreferenceRecord, ...],
        tuple[MissingDistinguishingRequirement, ...],
        tuple[QueryUnderstandingIssue, ...],
    ]:
        required: list[ExtractedConstraintRecord] = []
        negative: list[ExtractedNegativeConstraintRecord] = []
        soft: list[ExtractedSoftPreferenceRecord] = []
        missing: list[MissingDistinguishingRequirement] = []
        issues: list[QueryUnderstandingIssue] = []

        for match in _CAPACITY_REQUIRED_SCAN.finditer(raw_text):
            cap = normalize_capacity_token(match.group(0))
            if cap is None:
                continue
            if _inside_soft_or_negative(raw_text, match.start()):
                continue
            required.append(
                _required_record(
                    raw_text,
                    match.start(),
                    match.end(),
                    CANONICAL_CAPACITY,
                    cap,
                    match.group(0),
                )
            )

        for match in _INTERFACE_REQUIRED_SCAN.finditer(raw_text):
            iface = normalize_interface_token(match.group(0))
            if iface is None:
                continue
            if _inside_soft_or_negative(raw_text, match.start()):
                continue
            required.append(
                _required_record(
                    raw_text,
                    match.start(),
                    match.end(),
                    CANONICAL_INTERFACE,
                    iface,
                    match.group(0),
                )
            )

        for match in _MEMORY_TYPE_PATTERN.finditer(raw_text):
            mem = normalize_memory_type_token(match.group(0))
            if mem is None:
                continue
            if _inside_soft_or_negative(raw_text, match.start()):
                continue
            required.append(
                _required_record(
                    raw_text,
                    match.start(),
                    match.end(),
                    CANONICAL_MEMORY_TYPE,
                    mem,
                    match.group(0),
                )
            )

        if normalize_ecc_required(raw_text):
            ecc_match = re.search(r"\bECC\b", raw_text, re.IGNORECASE)
            if ecc_match and not _inside_soft_or_negative(raw_text, ecc_match.start()):
                required.append(
                    _required_record(
                        raw_text,
                        ecc_match.start(),
                        ecc_match.end(),
                        CANONICAL_ECC,
                        "true",
                        ecc_match.group(0),
                    )
                )

        for match in _NEGATIVE_INTERFACE.finditer(raw_text):
            token = match.group("token")
            iface = normalize_interface_token(token)
            if iface is None:
                continue
            negative.append(
                _negative_record(raw_text, match.start(), match.end(), CANONICAL_INTERFACE, iface, token)
            )

        for match in _NEGATIVE_MEMORY.finditer(raw_text):
            mem = normalize_memory_type_token(match.group("token"))
            if mem is None:
                continue
            negative.append(
                _negative_record(
                    raw_text,
                    match.start(),
                    match.end(),
                    CANONICAL_MEMORY_TYPE,
                    mem,
                    match.group("token"),
                )
            )

        for match in _NEGATIVE_CAPACITY.finditer(raw_text):
            cap = normalize_capacity_token(match.group("cap"))
            if cap is None:
                continue
            negative.append(
                _negative_record(raw_text, match.start(), match.end(), CANONICAL_CAPACITY, cap, match.group("cap"))
            )

        for match in _SOFT_CAPACITY.finditer(raw_text):
            cap = normalize_capacity_token(match.group("cap"))
            if cap is None:
                continue
            soft.append(
                ExtractedSoftPreferenceRecord(
                    preference=StructuredAttributeConstraint(
                        attribute_name=CANONICAL_CAPACITY,
                        operator=StructuredConstraintOperator.EQUALS,
                        value=cap,
                    ),
                    source_span=_span(raw_text, match.start(), match.end()),
                    certainty=ExtractionCertainty.DIRECT,
                    raw_value=match.group("cap"),
                    normalized_value=cap,
                )
            )

        for idx, match in enumerate(_MISSING_DISTINGUISHING.finditer(raw_text)):
            attr_raw = match.group("a").casefold()
            if attr_raw in ("sata", "nvme"):
                attr_name = CANONICAL_INTERFACE
            else:
                attr_name = CANONICAL_MEMORY_TYPE
            missing.append(
                MissingDistinguishingRequirement(
                    attribute_name=attr_name,
                    origin=MissingRequirementOrigin.USER,
                    requirement_id=f"user-missing-{idx}",
                )
            )

        if _has_conflicting_required(required):
            issues.append(
                QueryUnderstandingIssue(
                    code=QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT,
                    detail="conflicting required attribute values",
                )
            )

        required_deduped = _dedupe_required(required)

        return (
            tuple(required_deduped),
            tuple(_dedupe_negative(negative)),
            tuple(soft),
            tuple(missing),
            tuple(issues),
        )


_CAPACITY_REQUIRED_SCAN = _CAPACITY_PATTERN
_INTERFACE_REQUIRED_SCAN = re.compile(
    r"\b(NVMe|SATA|PCIe|USB-C)\b",
    re.IGNORECASE,
)


def _inside_soft_or_negative(text: str, position: int) -> bool:
    window_start = max(0, position - 24)
    prefix = text[window_start:position].casefold()
    return any(
        needle in prefix
        for needle in (
            "not ",
            "without ",
            "no ",
            "except ",
            "prefer ",
            "ideally ",
            "preferably ",
        )
    )


def _required_record(
    raw_text: str,
    start: int,
    end: int,
    attribute_name: str,
    value: str,
    raw_value: str,
) -> ExtractedConstraintRecord:
    return ExtractedConstraintRecord(
        constraint=StructuredAttributeConstraint(
            attribute_name=attribute_name,
            operator=StructuredConstraintOperator.EQUALS,
            value=value,
        ),
        source_span=_span(raw_text, start, end),
        certainty=ExtractionCertainty.DETERMINISTIC,
        raw_value=raw_value,
        normalized_value=value,
    )


def _negative_record(
    raw_text: str,
    start: int,
    end: int,
    attribute_name: str,
    value: str,
    raw_value: str,
) -> ExtractedNegativeConstraintRecord:
    return ExtractedNegativeConstraintRecord(
        constraint=NegativeAttributeConstraint(
            attribute_name=attribute_name,
            operator=StructuredConstraintOperator.EQUALS,
            excluded_value=value,
        ),
        source_span=_span(raw_text, start, end),
        certainty=ExtractionCertainty.DIRECT,
        raw_value=raw_value,
        normalized_value=value,
    )


def _dedupe_required(
    records: list[ExtractedConstraintRecord],
) -> list[ExtractedConstraintRecord]:
    by_attr: dict[str, ExtractedConstraintRecord] = {}
    for item in sorted(records, key=lambda r: r.source_span.start_offset):
        key = item.constraint.attribute_name.casefold()
        by_attr[key] = item
    return [by_attr[k] for k in sorted(by_attr)]


def _dedupe_negative(
    records: list[ExtractedNegativeConstraintRecord],
) -> list[ExtractedNegativeConstraintRecord]:
    seen: set[tuple[str, str]] = set()
    ordered: list[ExtractedNegativeConstraintRecord] = []
    for item in sorted(records, key=lambda r: r.source_span.start_offset):
        key = (item.constraint.attribute_name.casefold(), item.constraint.excluded_value.casefold())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _has_conflicting_required(records: list[ExtractedConstraintRecord]) -> bool:
    values_by_attr: dict[str, set[str]] = {}
    for item in records:
        key = item.constraint.attribute_name.casefold()
        values_by_attr.setdefault(key, set()).add(item.constraint.value.casefold())
    return any(len(values) > 1 for values in values_by_attr.values())
