# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    LLAMA_3_1 = "llama3.1:latest"
    GPT_OSS_20B = "gpt-oss:20b"


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.LLAMA_3_1)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int
    