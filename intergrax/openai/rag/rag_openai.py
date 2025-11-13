# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from openai import OpenAI
from openai.types.vector_store import VectorStore
from typing import List, Iterable
from pathlib import Path
from tqdm.auto import tqdm
import time
from dotenv import load_dotenv
load_dotenv()

class IntergraxRagOpenAI:

    def __init__(self, client: OpenAI, vector_store_id:str):
        self.client = client
        self.vector_store_id = vector_store_id


    def rag_prompt(self) -> str:
        prompt = """
        ROLE (STRICT RAG)

        You are a Knowledge Retrieval Assistant. Your ONLY allowed source of truth is the content retrieved from documents via the file_search tool (vector store).  
        You MUST NOT use general knowledge, outside facts, assumptions, or world knowledge.

        PURPOSE

        Answer the user’s questions using ONLY the retrieved document fragments.  
        Provide accurate, thorough, source-backed answers.

        WORKFLOW (MANDATORY, STEP-BY-STEP)

        1. Understand the question.  
        - If multi-part: split into sub-questions and address each one.

        2. Retrieve context.  
        - Use file_search.  
        - Perform multiple differently-phrased queries if needed.  
        - Ensure you have enough coverage.

        3. Verify consistency.  
        - Compare fragments.  
        - If contradictions appear: explicitly describe them and list possible interpretations (each with source reference).

        4. Answer.  
        - Write concise conclusions.  
        - Then provide expanded explanation (definitions, context, consequences).  
        - ALL content must come from cited fragments.

        5. Cite sources.  
        - After each important claim add a parenthetical reference:  
            (Source: file_name, p. X) or (Source: file_name, section Y).  
        - For long answers: add a final “Sources” section.  
        - Use direct quotes only when truly necessary and keep them short.

        UNCERTAINTY RULES

        If the documents do NOT contain enough information:  
        - Say explicitly: “Based on the available documents, I cannot fully answer X.”  
        - Specify what is missing (section name, document type, etc.).  
        - Suggest concrete search phrases or additional documents.

        You MUST NOT:  
        - invent information  
        - speculate  
        - rely on prior knowledge  
        - fill gaps with assumptions  

        If you infer something from the provided fragments, label it clearly as:  
        “Conclusion based on sources.”

        RESPONSE STYLE

        1. Start with a short, 2–4 sentence summary.  
        2. Then provide detailed explanation:  
        - step-by-step reasoning  
        - bullet lists  
        - small headings  
        3. Use precise terminology, no generalities or abstract phrasing.  
        4. For procedures or algorithms: produce a checklist or pseudo-procedure.  
        5. For numeric values: provide exact numbers and cite sources.

        OUTPUT FORMAT

        Summary  
        Detailed explanation (with inline citations)  
        Sources (file name + page/section)

        PROHIBITED ACTIONS (ABSOLUTE)

        - Do not use ANY information outside the retrieved documents.  
        - Do not rely on common knowledge, intuition, or the internet.  
        - Do not hide uncertainty.  
        - Do not strengthen or reinterpret claims beyond what is written.

        EXAMPLES OF REFERENCES

        “... according to the process definition (Source: Specification_Process_A.pdf, p. 12) ...”  
        “... non-functional requirement: availability 99.9% (Source: System_Requirements.docx, section 3.2) ...”
        """
        return prompt


    def ensure_vector_store_exists(self)-> VectorStore:
        """Retrieve vector store by its id"""
        try:
            vs = self.client.vector_stores.retrieve(self.vector_store_id)
            return vs
        except Exception as e:
            print(f"Error while retreiving vector store: {e}")
            raise
        

    def clear_vector_store_and_storage(self)->None:
        """Delete all files loaded into vectorstore"""

        files_page = self.client.vector_stores.files.list(
            vector_store_id=self.vector_store_id,
            limit=100
        )

        file_ids: List[str] = [f.id for f in files_page.data]

        if not file_ids:
            print("No files in vector store")
            return
        else:
            print(f"Found {len(file_ids)} files in vector store")

        next_page = getattr(files_page, "has_more", False)
        cursor = getattr(files_page, "last_id", None)

        with tqdm(desc=f"Loading files from VS {self.vector_store_id}", unit="page", leave=False) as pbar:

            while next_page and cursor:
                page = self.client.vector_stores.files.list(
                    vector_store_id=self.vector_store_id,
                    after=cursor,
                    limit=100
                )

                file_ids.extend([f.id for f in page.data])
                next_page = getattr(page, "has_more", False)
                cursor = getattr(page, "last_id", None)
                pbar.update(1)

        if file_ids:
            with tqdm(desc=f"Deleting files from VS {self.vector_store_id}", unit="page", leave=False, total=len(file_ids)) as pbar:
                for fid in file_ids:
                    try:                        
                        self.client.vector_stores.files.delete(
                            vector_store_id=self.vector_store_id,
                            file_id=fid
                        )

                        pbar.update(1)
                    except Exception as e:
                        print(f"Error while deleting file from vector store: {fid}: {e}")
                        continue
            
            with tqdm(desc=f"Deleting files from storage", unit="page", leave=False, total=len(file_ids)) as pbar:
                for fid in file_ids:
                    try:
                        self.client.files.delete(file_id=fid)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error while deleting file: {fid}: {e}")
                        continue


    def upload_folder_to_vector_store(self, folder: str | Path, patterns: Iterable = ("*.pdf", "*.txt", "*.doc", "*.docx"))->None:
        folder = Path(folder)
        if not folder.exists():
            print(f"Directory {folder} not exists.")
            return
        
        paths : List[Path] = []

        for pat in patterns:
            paths.extend(folder.glob(pat))

        if not paths:
            print(f"No files found in folder: {folder}")
            return
        
        print(f"Found {len(paths)} files.")
        print("\n".join([f.name for f in paths]))
        
        with tqdm(desc=f"Uploading files to {self.vector_store_id}", unit="page", leave=False, total=len(paths)) as pbar:
            for p in paths:
                with open(p, "rb") as file:                    
                    up = self.client.files.create(file=file, purpose="user_data")
                
                print(f"Uploaded {p.name} (id={up.id}) - checking availability...")

                while True:
                    print(f"Checking status: {p}")
                    f_info = self.client.files.retrieve(up.id)

                    if f_info.status == "uploaded":
                       print(f"File {p.name} is not ready - waiting")
                       time.sleep(5)
                       continue     

                    if f_info.status == "error":
                        print(f"File {p.name} failed to upload (storage error).")
                        break
                    
                    if f_info.status == "processed":
                        print(f"File {p.name} processed")
                        break                
                
                link = self.client.vector_stores.files.create(
                    vector_store_id=self.vector_store_id,
                    file_id=up.id,
                )

                pbar.update(1)        
                    
                    
        print(f"Upload completed ({len(paths)}).")
        
    
    def run(self, question: str, model:str="gpt-5-mini", instructions:str=None, n_results:int=10)->str:

        if not instructions:
            instructions = self.rag_prompt()

        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=question,
            tools=[
                {
                    "type":"file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": n_results,
                    "ranking_options": {
                        "ranker": "auto",
                        "score_threshold": 0.2,
                    }
                }
            ],            
        )

        return response.output_text

        
