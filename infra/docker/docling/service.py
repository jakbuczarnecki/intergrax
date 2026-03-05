from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Request
from docling.document_converter import DocumentConverter


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.converter = DocumentConverter()
    try:
        yield
    finally:
        # Docling DocumentConverter does not expose an explicit close() contract.
        # Keeping this for future-proofing without inventing APIs.
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse(request: Request, file: UploadFile):
    data = await file.read()
    converter: DocumentConverter = request.app.state.converter
    result = converter.convert(data)
    return {"text": result.document.export_to_markdown()}