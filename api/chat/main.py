# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import shutil
import uuid
import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from .tools.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from .tools.db_utils import insert_application_logs, get_chat_history, get_all_documents, delete_document_record, insert_document_record
from .tools.chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from .tools.rag_pipeline import get_answerer,set_history
logging.basicConfig(filename="app.log", level=logging.INFO)

app = FastAPI()

def _history_pairs_from_db(session_id: str):
    raw = get_chat_history(session_id)  # [{'role':'human','content':...}, {'role':'ai','content':...}, ...]
    pairs = []
    buf_user = None
    for m in raw:
        if m["role"] in ("human", "user"):
            buf_user = m["content"]
        elif m["role"] in ("ai", "assistant"):
            if buf_user is not None:
                pairs.append((buf_user, m["content"]))
                buf_user = None
    return pairs


@app.post(path="/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    model_name = query_input.model.value
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {model_name}")

    answerer = get_answerer(model_name)
    pairs = _history_pairs_from_db(session_id)
    set_history(answerer, pairs)

    res = answerer.run(
        query_input.question,
        where=None,
        stream=False,
        summarize=False,
    )
    answer = res["answer"]

    insert_application_logs(session_id, query_input.question, answer, model_name)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)

        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {
                "message": f"File {file.filename} has been successfully uploaded and indexed",
                "file_id": file_id
            }
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")

    except Exception as e:
        logging.error(f"Uploading and indexing document error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error while uploading/indexing the document.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {
                "message": f"Successfully deleted document with file_id {request.file_id} from system."
            }
        else:
            return {
                "error": f"Deleted from Chroma but failed to delete document with file_id: {request.file_id} from the database"
            }
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete file_id {request.file_id} from Chroma.")
