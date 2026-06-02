# © Artur Czarnecki. All rights reserved.

from pydantic import BaseModel, Field


class MlPredictInput(BaseModel):
    artifact_id: str = "ml.stub.classifier"
    adapter_slug: str = "sklearn_classifier"
    features: dict[str, float] = Field(default_factory=dict)


class MlPredictOutput(BaseModel):
    request_id: str
    predictions: dict[str, float] = Field(default_factory=dict)
