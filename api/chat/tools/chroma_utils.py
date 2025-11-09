# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List
from langchain_core.documents import Document
from .rag_pipeline import load_and_split_documents,index_document_to_vectorstore,delete_by_file_id

def load_and_split_documents(file_path: str) -> List[Document]:
    return load_and_split_documents(file_path)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    # wpinamy RAG ingest i dopisujemy file_id w metadanych (robi to pipeline)
    return index_document_to_vectorstore(file_path, file_id)

def delete_doc_from_chroma(file_id: int) -> bool:
    try:
        delete_by_file_id(file_id)
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id}: {e}")
        return False
