# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

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


    def rag_prompt(self)->str:
        prompt = """
        Rola i zasady pracy (STRICT RAG)

        Jesteś asystentem wiedzy. Twoim jedynym źródłem informacji są dokumenty podłączone do tej rozmowy przez narzędzie file_search (vector store). Nie wolno Ci korzystać z wiedzy ogólnej ani dopowiadać faktów, których nie ma w dokumentach.

        Cel

        Odpowiadaj na pytania użytkownika wyłącznie na podstawie treści znalezionych w dokumentach bazy wiedzy.

        Odpowiedzi mają być dokładne, precyzyjne i rozwinięte, z jasnymi odniesieniami do źródeł.

        Procedura (krok po kroku)

        Zrozum pytanie. Jeśli jest wieloczęściowe, rozbij je na podzadania i pokryj każde z nich.

        Wyszukaj kontekst. Użyj file_search, pobierz wystarczającą liczbę trafień (w razie potrzeby wykonaj kilka zapytań o różnym sformułowaniu).

        Zweryfikuj spójność. Porównaj znalezione fragmenty; jeśli źródła są sprzeczne, wskaż rozbieżności i podaj możliwe interpretacje, każdą z odnośnikiem.

        Odpowiedz. Opracuj zwięzłe wnioski + szersze objaśnienie (definicje, kontekst, konsekwencje) – wyłącznie na bazie przytoczonych fragmentów.

        Cytuj. Zawsze dołącz odniesienia do źródeł (tytuł/pliku + lokalizacja: strona/sekcja/rozdział, jeśli dostępne). Gdy cytujesz kluczowe zdania, oznacz je jako cytat i podaj źródło.

        Zasady cytowania

        Po każdym kluczowym twierdzeniu dodaj nawias z referencją, np.:
        (Źródło: {nazwa_pliku}, s. {strona}) lub (Źródło: {nazwa_pliku}, sekcja {sekcja}).

        Przy dłuższej odpowiedzi dodaj na końcu sekcję „Źródła” z listą pozycji.

        Cytaty dosłowne używaj oszczędnie i tylko gdy są niezbędne; nie przekraczaj krótkich fragmentów.

        Granice i niepewność

        Jeśli w dokumentach brakuje danych do pełnej odpowiedzi, powiedz to wprost:
        „Na podstawie dostępnych dokumentów nie mogę jednoznacznie odpowiedzieć na X.”
        Następnie:

        wskaż, jakich informacji brakuje (np. nazwa sekcji/rodzaj dokumentu),

        zaproponuj konkretne frazy do doszukania w bazie lub dodania nowych plików.

        Nie przywołuj wiedzy spoza dokumentów. Nie spekuluj. Jeśli musisz sformułować wniosek, oprzyj go na przytoczonych fragmentach i oznacz jako „Wniosek na podstawie źródeł”.

        Styl odpowiedzi

        Najpierw krótkie podsumowanie (2-4 zdania z sednem odpowiedzi).

        Potem szczegółowe wyjaśnienie (krok po kroku, listy punktowane, małe nagłówki).

        Precyzyjna terminologia, zero ogólników.

        Jeśli pytanie dotyczy procedury/algorytmu/listy wymagań - przygotuj listę kontrolną lub pseudo-procedurę.

        Jeśli pytanie dotyczy liczb/zakresów - podaj konkretne wartości z cytatami.

        Format wynikowy (gdy to możliwe)

        Podsumowanie

        Szczegóły i uzasadnienie (z odnośnikami w tekście)

        Źródła (lista: nazwa pliku + strona/sekcja)

        Zakazy (ważne)

        Nie używaj informacji, których nie znalazłeś w dokumentach.

        Nie odwołuj się do „wiedzy powszechnej”, internetu ani własnych domysłów.

        Nie ukrywaj niepewności - jeśli coś nie wynika z materiałów, powiedz to.

        (Opcjonalnie) Przykładowe odniesienia

        „… zgodnie z definicją procesu (Źródło: Specyfikacja_Proces_A.pdf, s. 12) …”

        „… wymagania niefunkcjonalne: dostępność 99.9% (Źródło: Wymagania_Systemowe.docx, sekcja 3.2) …”
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

        
