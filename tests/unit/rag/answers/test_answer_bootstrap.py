# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.rag.answers.contracts.answer_engine import AnswerEngine
from intergrax.rag.answers.pipeline.answer_pipeline import AnswerPipeline
from intergrax.rag.answers.bootstrap.answer_bootstrap import (
    create_default_answer_engine,
    create_default_answer_pipeline,
)
from tests._support.builder import FakeLLMAdapter


pytestmark = pytest.mark.unit


def test_create_default_answer_pipeline_returns_answer_pipeline():
    pipeline = create_default_answer_pipeline()

    assert isinstance(pipeline, AnswerPipeline)


def test_create_default_answer_engine_returns_answer_engine():
    engine = create_default_answer_engine(
        llm=FakeLLMAdapter(),
    )

    assert isinstance(engine, AnswerEngine)