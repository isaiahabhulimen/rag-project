import chromadb

from models import model
from config import collection_name, chunk_size, chunk_overlap
from indexer import index_books
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    print("Starting book ingestion...")

    client = chromadb.PersistentClient(
        path="database"
    )

    text_collection = client.get_or_create_collection(
        name=f"{collection_name}_text"
    )

    image_collection = client.get_or_create_collection(
        name=f"{collection_name}_images"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    index_books(
        text_collection,
        image_collection,
        model,
        splitter
    )

    print("Book ingestion completed.")


if __name__ == "__main__":
    main()