import os
import time

import pdfplumber

from utils import get_file_hash
from config import (
    embedding_batch_size,
    book_folder,
    semantic_chunk_threshold,
    min_chunk_characters,
    max_chunk_characters,
)
from preprocessor import clean_text
from vision import describe_image
from semantic_chunker import semantic_chunk


def index_books(text_collection, image_collection, model, splitter):
    book_files = os.listdir(book_folder)

    print(book_files)

    for book_name in book_files:

        if not book_name.endswith(".pdf"):
            continue

        pdf_path = os.path.join(book_folder, book_name)

        print(f"\n===== Processing: {book_name} =====")

        file_hash = get_file_hash(pdf_path)

        existing_book = text_collection.get(
            where={"source": book_name}
        )

        stored_hash = (
            existing_book["metadatas"][0]["file_hash"]
            if existing_book["ids"]
            else None
        )

        print(stored_hash)

        if existing_book["ids"]:

            if stored_hash == file_hash:
                print("Book unchanged, skipping indexing")
                continue

            print("Book has changed. Updating index...")

            text_collection.delete(
                where={"source": book_name}
            )

            image_collection.delete(
                where={"source": book_name}
            )

        indexing_start = time.perf_counter()

        text_batch = []
        image_batch = []

        text_index = 0
        image_index = 0

        total_text_chunks = 0
        total_image_chunks = 0

        try:

            with pdfplumber.open(pdf_path) as pdf:

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1
                ):

                    # -------------------------
                    # TEXT PROCESSING
                    # -------------------------

                    page_text = clean_text(
                        page.extract_text()
                    )

                    if page_text:

                        page_chunks = semantic_chunk(
                            page_text,
                            model,
                            semantic_chunk_threshold,
                            min_chunk_characters,
                            max_chunk_characters
                        )

                        for chunk in page_chunks:

                            text_batch.append({
                                "page": page_number,
                                "text": chunk,
                                "type": "text"
                            })

                            total_text_chunks += 1

                            if len(text_batch) >= embedding_batch_size:

                                text_index = _store_text_batch(
                                    text_collection,
                                    model,
                                    text_batch,
                                    book_name,
                                    file_hash,
                                    text_index
                                )

                                text_batch = []

                                print(
                                    f"> Indexed "
                                    f"{total_text_chunks} text chunks"
                                )

                    # -------------------------
                    # IMAGE PROCESSING
                    # -------------------------

                    for img_data in page.images:

                        img_path = (
                            f"temp_img_{book_name}_"
                            f"{page_number}_"
                            f"{image_index}.png"
                        )

                        try:

                            img_bytes = (
                                img_data["stream"].get_data()
                            )

                            with open(
                                img_path,
                                "wb"
                            ) as img_file:

                                img_file.write(img_bytes)

                            description = describe_image(
                                img_path
                            )

                            image_batch.append({
                                "page": page_number,
                                "text": (
                                    "[Image description: "
                                    f"{description}]"
                                ),
                                "type": "image"
                            })

                            total_image_chunks += 1
                            image_index += 1

                            if (
                                len(image_batch)
                                >= embedding_batch_size
                            ):

                                image_index = _store_image_batch(
                                    image_collection,
                                    model,
                                    image_batch,
                                    book_name,
                                    file_hash,
                                    image_index
                                )

                                image_batch = []

                                print(
                                    f"> Indexed "
                                    f"{total_image_chunks} "
                                    f"image chunks"
                                )

                        finally:

                            if os.path.exists(img_path):
                                os.remove(img_path)

            # -------------------------
            # FLUSH REMAINING TEXT
            # -------------------------

            if text_batch:

                _store_text_batch(
                    text_collection,
                    model,
                    text_batch,
                    book_name,
                    file_hash,
                    text_index
                )

                print(
                    f"> Indexed "
                    f"{total_text_chunks} text chunks"
                )

            # -------------------------
            # FLUSH REMAINING IMAGES
            # -------------------------

            if image_batch:

                _store_image_batch(
                    image_collection,
                    model,
                    image_batch,
                    book_name,
                    file_hash,
                    image_index
                )

                print(
                    f"> Indexed "
                    f"{total_image_chunks} image chunks"
                )

            indexing_end = time.perf_counter()

            indexing_time = (
                indexing_end - indexing_start
            )

            print(
                f"Indexed {total_text_chunks} "
                f"text chunks and "
                f"{total_image_chunks} image chunks"
            )

            print(
                f"Indexing completed in "
                f"{indexing_time:.2f} seconds."
            )

        except Exception:

            # Remove partially indexed data so that
            # the next ingestion attempt can retry
            # the book instead of treating it as complete.

            print(
                f"Indexing failed for {book_name}. "
                "Removing partial index..."
            )

            text_collection.delete(
                where={"source": book_name}
            )

            image_collection.delete(
                where={"source": book_name}
            )

            raise


def _store_text_batch(
    collection,
    model,
    batch,
    book_name,
    file_hash,
    start_index
):

    batch_texts = [
        chunk["text"]
        for chunk in batch
    ]

    embeddings = model.encode(
        batch_texts,
        batch_size=embedding_batch_size
    ).tolist()

    ids = []
    documents = []
    metadatas = []

    for offset, (chunk, embedding) in enumerate(
        zip(batch, embeddings)
    ):

        index = start_index + offset

        ids.append(
            f"{book_name}_text_{index}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "page": chunk["page"],
            "source": book_name,
            "file_hash": file_hash,
            "type": "text"
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=[
            embedding
            for embedding in embeddings
        ],
        metadatas=metadatas
    )

    return start_index + len(batch)


def _store_image_batch(
    collection,
    model,
    batch,
    book_name,
    file_hash,
    start_index
):

    batch_texts = [
        chunk["text"]
        for chunk in batch
    ]

    embeddings = model.encode(
        batch_texts,
        batch_size=embedding_batch_size
    ).tolist()

    ids = []
    documents = []
    metadatas = []

    for offset, (chunk, embedding) in enumerate(
        zip(batch, embeddings)
    ):

        index = start_index + offset

        ids.append(
            f"{book_name}_image_{index}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "page": chunk["page"],
            "source": book_name,
            "file_hash": file_hash,
            "type": "image"
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=[
            embedding
            for embedding in embeddings
        ],
        metadatas=metadatas
    )

    return start_index + len(batch)