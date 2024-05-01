from pt_embeddings_file_one import Get_openai_embedding
from langchain.vectorstores.chroma import Chroma
from rich.console import Console
import os
from pt_populate_database import main

class DB_Connect:
    def __init__(self, db_location, docs_location):
        # db location
        self.db_location = db_location

        # docs location
        self.docs_location = docs_location

        # db object
        self.db = Chroma(
            persist_directory=self.db_location,
            embedding_function=Get_openai_embedding("text-embedding-3-small")
        )

    def query_rag(self, query_text: str):
        # this should be a string.
        results = self.db.similarity_search_with_score(query_text, k=3)
        return results
    
    def refreshDatabase(self):
        # we are going to use the main function from the populate database file.
        refresh_status = main(self.db_location, self.docs_location)
        return refresh_status

