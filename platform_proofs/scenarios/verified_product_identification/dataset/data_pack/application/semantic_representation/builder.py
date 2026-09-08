"""Priority-based bounded semantic text builder."""

from __future__ import annotations

import re
from collections.abc import Sequence

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    classify_wdc_identifier_type,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.contracts import (
    RepresentationSectionKind,
    SemanticRepresentationBuildOutput,
    SemanticRepresentationPolicy,
    SemanticRepresentationResult,
    SemanticSection,
    TruncationStrategy,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.ports import (
    CharacterRatioTokenEstimator,
    TokenEstimatorPort,
)

_PRIORITY_IDENTITY = 0
_PRIORITY_TITLE = 1
_PRIORITY_ATTRIBUTES = 2
_PRIORITY_DESCRIPTION = 3
_PRIORITY_SPECS = 4

_HIGH_PRIORITY_ATTRIBUTE_KEYS = frozenset(
    {
        "capacity",
        "compatibility",
        "dimension",
        "dimensions",
        "gtin",
        "interface",
        "manufacturer",
        "model",
        "mpn",
        "part number",
        "partnumber",
        "power",
        "sku",
        "size",
        "voltage",
        "weight",
        "width",
        "height",
        "length",
        "depth",
    }
)
_MEDIUM_PRIORITY_ATTRIBUTE_KEYS = frozenset(
    {
        "color",
        "colour",
        "material",
        "warranty",
        "finish",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")


class SemanticRepresentationBuilder:
    """Build bounded semantic text from one catalog record using priority compression."""

    def __init__(
        self,
        policy: SemanticRepresentationPolicy,
        token_estimator: TokenEstimatorPort | None = None,
    ) -> None:
        if policy.truncation_strategy is not TruncationStrategy.PRIORITY_COMPRESSION:
            msg = f"unsupported truncation strategy: {policy.truncation_strategy}"
            raise ValueError(msg)
        self._policy = policy
        self._token_estimator = token_estimator or CharacterRatioTokenEstimator()

    def build(
        self,
        source_offer: WdcSourceOffer,
        *,
        source_ref: SourceRecordRef,
    ) -> SemanticRepresentationBuildOutput:
        sections = self._extract_sections(source_offer)
        original_text = _join_sections(sections)
        original_character_count = len(original_text)

        compressed_sections, truncated = self._apply_budget(sections)
        representation_text = _join_sections(compressed_sections)
        final_character_count = len(representation_text)
        estimated_tokens = self._token_estimator.estimate_tokens(representation_text)

        result = SemanticRepresentationResult(
            source_ref=source_ref,
            original_character_count=original_character_count,
            final_character_count=final_character_count,
            estimated_tokens=estimated_tokens,
            truncated=truncated or final_character_count < original_character_count,
            preserved_field_count=len(compressed_sections),
            representation_text=representation_text,
        )
        return SemanticRepresentationBuildOutput(
            result=result,
            final_sections=tuple(compressed_sections),
        )

    def _extract_sections(self, source_offer: WdcSourceOffer) -> tuple[SemanticSection, ...]:
        sections: list[SemanticSection] = []

        identity_lines = _build_identity_lines(source_offer)
        if identity_lines:
            sections.append(
                SemanticSection(
                    kind=RepresentationSectionKind.IDENTITY,
                    source_field="identifiers",
                    text="\n".join(identity_lines),
                    priority=_PRIORITY_IDENTITY,
                )
            )

        if source_offer.title is not None and source_offer.title.strip():
            sections.append(
                SemanticSection(
                    kind=RepresentationSectionKind.TITLE,
                    source_field="title",
                    text=source_offer.title.strip(),
                    priority=_PRIORITY_TITLE,
                )
            )

        attribute_lines = _build_attribute_lines(source_offer.key_value_pairs)
        if attribute_lines:
            sections.append(
                SemanticSection(
                    kind=RepresentationSectionKind.ATTRIBUTES,
                    source_field="keyValuePairs",
                    text="\n".join(attribute_lines),
                    priority=_PRIORITY_ATTRIBUTES,
                )
            )

        if source_offer.description is not None and source_offer.description.strip():
            sections.append(
                SemanticSection(
                    kind=RepresentationSectionKind.DESCRIPTION,
                    source_field="description",
                    text=source_offer.description.strip(),
                    priority=_PRIORITY_DESCRIPTION,
                )
            )

        if source_offer.spec_table_content is not None and source_offer.spec_table_content.strip():
            sections.append(
                SemanticSection(
                    kind=RepresentationSectionKind.SPECS,
                    source_field="specTableContent",
                    text=source_offer.spec_table_content.strip(),
                    priority=_PRIORITY_SPECS,
                )
            )

        return tuple(sections)

    def _apply_budget(
        self,
        sections: tuple[SemanticSection, ...],
    ) -> tuple[tuple[SemanticSection, ...], bool]:
        if not sections:
            return (), False

        working = list(sections)
        truncated = False

        while not self._within_budget(working):
            if not self._shrink_lowest_priority_section(working):
                break
            truncated = True

        return tuple(working), truncated

    def _within_budget(self, sections: list[SemanticSection]) -> bool:
        text = _join_sections(sections)
        if self._policy.max_characters is not None and len(text) > self._policy.max_characters:
            return False
        if self._policy.max_tokens is not None:
            estimated = self._token_estimator.estimate_tokens(text)
            if estimated > self._policy.max_tokens:
                return False
        return True

    def _shrink_lowest_priority_section(self, sections: list[SemanticSection]) -> bool:
        if not sections:
            return False

        target_index = max(range(len(sections)), key=lambda index: sections[index].priority)
        target = sections[target_index]

        if target.priority <= _PRIORITY_TITLE:
            return self._trim_section_text(sections, target_index, reduction_ratio=0.10)

        if target.kind is RepresentationSectionKind.ATTRIBUTES:
            return self._drop_lowest_priority_attribute_line(sections, target_index)

        return self._trim_section_text(sections, target_index, reduction_ratio=0.25)

    def _drop_lowest_priority_attribute_line(
        self,
        sections: list[SemanticSection],
        section_index: int,
    ) -> bool:
        section = sections[section_index]
        lines = section.text.split("\n")
        if len(lines) <= 1:
            return self._trim_section_text(sections, section_index, reduction_ratio=0.25)

        scored_lines = sorted(
            enumerate(lines),
            key=lambda item: (_attribute_line_priority(item[1]), item[0]),
            reverse=True,
        )
        drop_index = scored_lines[-1][0]
        remaining = [line for index, line in enumerate(lines) if index != drop_index]
        if not remaining:
            sections.pop(section_index)
            return True
        sections[section_index] = SemanticSection(
            kind=section.kind,
            source_field=section.source_field,
            text="\n".join(remaining),
            priority=section.priority,
        )
        return True

    def _trim_section_text(
        self,
        sections: list[SemanticSection],
        section_index: int,
        *,
        reduction_ratio: float,
    ) -> bool:
        section = sections[section_index]
        if not section.text:
            sections.pop(section_index)
            return True

        target_length = max(1, int(len(section.text) * (1.0 - reduction_ratio)))
        if self._policy.max_characters is not None:
            current_total = len(_join_sections(sections))
            overflow = current_total - self._policy.max_characters
            if overflow > 0:
                target_length = max(1, len(section.text) - overflow)

        trimmed = _trim_text_to_length(section.text, target_length)
        if not trimmed.strip():
            sections.pop(section_index)
            return True

        sections[section_index] = SemanticSection(
            kind=section.kind,
            source_field=section.source_field,
            text=trimmed,
            priority=section.priority,
        )
        return True


def _build_identity_lines(source_offer: WdcSourceOffer) -> list[str]:
    lines: list[str] = []
    if source_offer.brand is not None and source_offer.brand.strip():
        lines.append(f"brand: {source_offer.brand.strip()}")

    for entry in source_offer.identifiers:
        identifier_type = classify_wdc_identifier_type(entry.source_key)
        if identifier_type is None:
            continue
        label = identifier_type.value
        value = entry.source_value.strip()
        if value:
            lines.append(f"{label}: {value}")

    for pair in source_offer.key_value_pairs:
        normalized_key = pair.source_key.strip().casefold()
        if normalized_key in {"model", "manufacturer", "part number", "partnumber", "mpn"}:
            value = pair.source_value.strip()
            if value:
                lines.append(f"{pair.source_key}: {value}")

    return _deduplicate_lines(lines)


def _build_attribute_lines(
    key_value_pairs: tuple[object, ...],
) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    identity_keys = {"model", "manufacturer", "part number", "partnumber", "mpn"}
    for index, pair in enumerate(key_value_pairs):
        source_key = getattr(pair, "source_key", "")
        source_value = getattr(pair, "source_value", "")
        if not isinstance(source_key, str) or not isinstance(source_value, str):
            continue
        normalized_key = source_key.strip().casefold()
        if normalized_key in identity_keys:
            continue
        line = f"{source_key.strip()}: {source_value.strip()}"
        if not line.strip(":").strip():
            continue
        scored.append((_attribute_key_priority(normalized_key), index, line))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [line for _, _, line in scored]


def _attribute_key_priority(normalized_key: str) -> int:
    if normalized_key in _HIGH_PRIORITY_ATTRIBUTE_KEYS:
        return 3
    if normalized_key in _MEDIUM_PRIORITY_ATTRIBUTE_KEYS:
        return 2
    return 1


def _attribute_line_priority(line: str) -> int:
    key = line.split(":", 1)[0].strip().casefold()
    return _attribute_key_priority(key)


def _deduplicate_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        normalized = _WHITESPACE_RE.sub(" ", line.strip().casefold())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(line)
    return unique


def _join_sections(sections: Sequence[SemanticSection]) -> str:
    return "\n".join(section.text for section in sections)


def _trim_text_to_length(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]

    candidate = text[: max_length - 3]
    last_space = candidate.rfind(" ")
    if last_space > max_length // 2:
        candidate = candidate[:last_space]
    return candidate.rstrip() + "..."
