# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

MODALITY_SPEECH_IO = SkillManifest(
    skill_id="modality.speech_io",
    version="1.0.0",
    description="Speech modality: transcribe audio input and synthesize spoken responses.",
    tool_ids=("speech.transcribe", "speech.synthesize"),
    prompt_instruction_ids=("modality.speech_io.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("modality", "speech", "audio"),
)

MODALITY_VISION_OCR = SkillManifest(
    skill_id="modality.vision_ocr",
    version="1.0.0",
    description="Vision OCR pipeline: detect regions, run OCR, and preview parsed document structure.",
    tool_ids=("vision.ocr_regions", "vision.detect", "document.parse_preview"),
    prompt_instruction_ids=("modality.vision_ocr.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("modality", "vision", "ocr"),
)

MODALITY_VISION_SEGMENT = SkillManifest(
    skill_id="modality.vision_segment",
    version="1.0.0",
    description="Vision segmentation pipeline: segment regions, detect, and OCR.",
    tool_ids=("vision.segment", "vision.detect", "vision.ocr_regions"),
    prompt_instruction_ids=("modality.vision_segment.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("modality", "vision", "segment"),
)

