# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class FilesystemListInput(BaseModel):
    path: str = Field(..., min_length=1, description="Absolute directory path inside an allowlisted root.")
    limit: int = Field(default=200, ge=1, le=2000)


class FilesystemEntryOutput(BaseModel):
    name: str
    path: str
    is_dir: bool


class FilesystemListOutput(BaseModel):
    path: str
    entries: list[FilesystemEntryOutput] = Field(default_factory=list)
    total: int = 0


class FilesystemGlobInput(BaseModel):
    pattern: str = Field(..., min_length=1, description="Glob pattern relative to an allowlisted root.")
    root: str = Field(..., min_length=1, description="Allowlisted root directory for the glob.")
    limit: int = Field(default=200, ge=1, le=2000)


class FilesystemGlobOutput(BaseModel):
    root: str
    pattern: str
    paths: list[str] = Field(default_factory=list)
    total: int = 0


class FilesystemReadTextInput(BaseModel):
    path: str = Field(..., min_length=1, description="Absolute file path inside an allowlisted root.")
    max_bytes: int = Field(default=65536, ge=1, le=1_048_576)


class FilesystemReadTextOutput(BaseModel):
    path: str
    text: str
    truncated: bool = False
    size_bytes: int = 0


class FilesystemStatInput(BaseModel):
    path: str = Field(..., min_length=1, description="Absolute file or directory path inside an allowlisted root.")


class FilesystemStatOutput(BaseModel):
    path: str
    exists: bool
    is_dir: bool = False
    is_file: bool = False
    size_bytes: int = 0
    modified_at_utc: str = ""


class FilesystemWriteTextInput(BaseModel):
    path: str = Field(..., min_length=1, description="Absolute file path inside an allowlisted root.")
    content: str = Field(..., description="UTF-8 text content to write.")
    create_dirs: bool = Field(default=True, description="Create parent directories when missing.")
    max_bytes: int = Field(default=1_048_576, ge=1, le=1_048_576)


class FilesystemWriteTextOutput(BaseModel):
    path: str
    written: bool = False
    size_bytes: int = 0
    created: bool = False
