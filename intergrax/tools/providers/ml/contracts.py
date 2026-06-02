# © Artur Czarnecki. All rights reserved.

from pydantic import BaseModel, Field


class MlPredictInput(BaseModel):
    artifact_id: str = "ml.stub.classifier"
    adapter_slug: str = "sklearn_classifier"
    features: dict[str, float] = Field(default_factory=dict)


class MlPredictOutput(BaseModel):
    request_id: str
    predictions: dict[str, float] = Field(default_factory=dict)


class MlExplainInput(BaseModel):
    artifact_id: str = "ml.stub.classifier"
    adapter_slug: str = "sklearn_classifier"
    features: dict[str, float] = Field(default_factory=dict)


class MlExplainOutput(BaseModel):
    request_id: str
    predictions: dict[str, float] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)


class MlBatchPredictInput(BaseModel):
    artifact_id: str = "ml.stub.classifier"
    adapter_slug: str = "sklearn_classifier"
    feature_rows: list[dict[str, float]] = Field(default_factory=list)


class MlBatchPredictOutput(BaseModel):
    request_ids: list[str] = Field(default_factory=list)
    predictions: list[dict[str, float]] = Field(default_factory=list)
