import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from pt_embeddings_file_one import get_embeddings_function, Get_openai_embedding
from langchain.vectorstores.chroma import Chroma
from rich.console import Console

con = Console()

def main(CHROMA_PATH, DATA_PATH):
    try:
        # Check if the database should be cleared (using the --clear flag).
        parser = argparse.ArgumentParser()
        parser.add_argument("--reset", action="store_true", help="Reset the database.")
        args = parser.parse_args()
        if args.reset:
            print("✨ Clearing Database")
            clear_database()

        # Create (or update) the data store.
        documents = load_documents(DATA_PATH)
        chunks = split_documents(documents)
        result = add_to_chroma(chunks, CHROMA_PATH)
        return result

    except Exception as e:
        return f"DB refresh failed\n{e}"


def load_documents(DATA_PATH):
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], CHROMA_PATH):
    # Load the existing database.
    try: 
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embeddings_function()
        )
    except:
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=Get_openai_embedding("text-embedding-3-small")
        )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    con.log(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        return("✅ New documents to add")
    
    else:
        return("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database(CHROMA_PATH):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)



# main('temp_db', 'Docs')