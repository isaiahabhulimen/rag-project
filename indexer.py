import os
import time

import pdfplumber

from config import (
    embedding_batch_size,
    persistence_batch_size,
    semantic_chunk_threshold,
    min_chunk_characters,
    max_chunk_characters,
)

from preprocessor import clean_text
from semantic_chunker import semantic_chunk
from utils import get_file_hash
from vision import describe_image


def index_book(
    pdf_path,
    book_name,
    job_id,
    model,
    store_batch,
    start_page=1,
    text_index=0,
    image_index=0,
    checkpoint_callback=None,
    checkpoint_interval=10,
):

    print(f"\n===== Processing: {book_name} =====")

    file_hash = get_file_hash(pdf_path)

    indexing_start = time.perf_counter()

    text_batch = []
    image_batch = []

    text_persist_batch = []
    image_persist_batch = []

    image_file_index = 0

    total_text_chunks = text_index
    total_image_chunks = image_index

    try:

        with pdfplumber.open(pdf_path) as pdf:

            total_pages = len(pdf.pages)

            print(f"Total pages detected: " f"{total_pages}")

            if start_page > total_pages:

                print(
                    f"Start page {start_page} "
                    f"is beyond the final page "
                    f"{total_pages}."
                )

                if checkpoint_callback:

                    checkpoint_callback(total_pages, text_index, image_index)

                return {
                    "file_hash": file_hash,
                    "text_chunks": text_index,
                    "image_chunks": image_index,
                    "indexing_time": (time.perf_counter() - indexing_start),
                }

            for page_number, page in enumerate(pdf.pages, start=1):

                if page_number < start_page:
                    continue

                # -------------------------
                # TEXT PROCESSING
                # -------------------------

                raw_text = page.extract_text()

                if raw_text:
                    page_text = clean_text(raw_text)
                else:
                    page_text = ""

                if page_text:

                    page_chunks = semantic_chunk(
                        page_text,
                        model,
                        semantic_chunk_threshold,
                        min_chunk_characters,
                        max_chunk_characters,
                    )

                    for chunk in page_chunks:

                        text_batch.append(
                            {"page": page_number, "text": chunk, "type": "text"}
                        )

                        total_text_chunks += 1

                        if len(text_batch) >= embedding_batch_size:

                            items = _embed_text_batch(
                                text_batch,
                                model,
                                book_name,
                                job_id,
                                file_hash,
                                text_index,
                            )

                            text_index += len(items)

                            text_persist_batch.extend(items)

                            text_batch = []

                            if len(text_persist_batch) >= persistence_batch_size:

                                store_batch(
                                    "text", book_name, file_hash, text_persist_batch
                                )

                                print(
                                    f"Persisted text batch: "
                                    f"{len(text_persist_batch)} chunks "
                                    f"(through chunk {text_index})"
                                )

                                text_persist_batch = []

                # -------------------------
                # IMAGE PROCESSING
                # -------------------------

                for image_data in page.images:

                    image_path = (
                        f"temp_img_"
                        f"{job_id}_"
                        f"{page_number}_"
                        f"{image_file_index}.png"
                    )

                    image_file_index += 1

                    try:

                        image_bytes = image_data["stream"].get_data()

                        with open(image_path, "wb") as image_file:

                            image_file.write(image_bytes)

                        description = describe_image(image_path)

                        image_batch.append(
                            {
                                "page": page_number,
                                "text": ("[Image description: " f"{description}]"),
                                "type": "image",
                            }
                        )

                        total_image_chunks += 1

                        if len(image_batch) >= embedding_batch_size:

                            items = _embed_image_batch(
                                image_batch,
                                model,
                                book_name,
                                job_id,
                                file_hash,
                                image_index,
                            )

                            image_index += len(items)

                            image_persist_batch.extend(items)

                            image_batch = []

                            if len(image_persist_batch) >= persistence_batch_size:

                                store_batch(
                                    "image", book_name, file_hash, image_persist_batch
                                )

                                print(
                                    f"Persisted image batch: "
                                    f"{len(image_persist_batch)} chunks "
                                    f"(through chunk {image_index})"
                                )

                                image_persist_batch = []

                    finally:

                        if os.path.exists(image_path):
                            os.remove(image_path)

                # -------------------------
                # PAGE PROGRESS
                # -------------------------

                elapsed = time.perf_counter() - indexing_start

                print(
                    f"Page {page_number}/{total_pages} | "
                    f"Text chunks: {total_text_chunks} | "
                    f"Image chunks: {total_image_chunks} | "
                    f"Elapsed: {elapsed:.2f}s"
                )

                # -------------------------
                # CHECKPOINT
                # -------------------------

                if checkpoint_callback and page_number % checkpoint_interval == 0:

                    # Finish remaining embedding batches.

                    if text_batch:

                        items = _embed_text_batch(
                            text_batch, model, book_name, job_id, file_hash, text_index
                        )

                        text_index += len(items)

                        text_persist_batch.extend(items)

                        text_batch = []

                    if image_batch:

                        items = _embed_image_batch(
                            image_batch,
                            model,
                            book_name,
                            job_id,
                            file_hash,
                            image_index,
                        )

                        image_index += len(items)

                        image_persist_batch.extend(items)

                        image_batch = []

                    # Persist everything before checkpointing.

                    if text_persist_batch:

                        store_batch("text", book_name, file_hash, text_persist_batch)

                        print(
                            f"Checkpoint text flush: "
                            f"{len(text_persist_batch)} chunks"
                        )

                        text_persist_batch = []

                    if image_persist_batch:

                        store_batch("image", book_name, file_hash, image_persist_batch)

                        print(
                            f"Checkpoint image flush: "
                            f"{len(image_persist_batch)} chunks"
                        )

                        image_persist_batch = []

                    # Save checkpoint only after all work
                    # for this boundary has been persisted.

                    checkpoint_callback(page_number, text_index, image_index)

                    print(f"Checkpoint saved: " f"page {page_number}")

            # -------------------------
            # FINAL TEXT EMBEDDING BATCH
            # -------------------------

            if text_batch:

                items = _embed_text_batch(
                    text_batch, model, book_name, job_id, file_hash, text_index
                )

                text_index += len(items)

                text_persist_batch.extend(items)

            # -------------------------
            # FINAL IMAGE EMBEDDING BATCH
            # -------------------------

            if image_batch:

                items = _embed_image_batch(
                    image_batch, model, book_name, job_id, file_hash, image_index
                )

                image_index += len(items)

                image_persist_batch.extend(items)

            # -------------------------
            # FINAL TEXT PERSISTENCE
            # -------------------------

            if text_persist_batch:

                store_batch("text", book_name, file_hash, text_persist_batch)

                print(
                    f"Persisted final text batch: " f"{len(text_persist_batch)} chunks"
                )

            # -------------------------
            # FINAL IMAGE PERSISTENCE
            # -------------------------

            if image_persist_batch:

                store_batch("image", book_name, file_hash, image_persist_batch)

                print(
                    f"Persisted final image batch: "
                    f"{len(image_persist_batch)} chunks"
                )

            # -------------------------
            # FINAL CHECKPOINT
            # -------------------------

            if checkpoint_callback:

                checkpoint_callback(total_pages, text_index, image_index)

                print(f"Checkpoint saved: " f"page {total_pages}")

            indexing_time = time.perf_counter() - indexing_start

            print(
                f"Indexed {text_index} "
                f"text chunks and "
                f"{image_index} "
                f"image chunks"
            )

            print(f"Indexing completed in " f"{indexing_time:.2f} seconds.")

            return {
                "file_hash": file_hash,
                "text_chunks": text_index,
                "image_chunks": image_index,
                "indexing_time": indexing_time,
            }

    except Exception:

        print(f"Indexing failed for " f"{book_name}")

        raise


def _embed_text_batch(batch, model, book_name, job_id, file_hash, start_index):

    texts = [chunk["text"] for chunk in batch]

    embeddings = model.encode(texts, batch_size=embedding_batch_size).tolist()

    items = []

    for offset, (chunk, embedding) in enumerate(zip(batch, embeddings)):

        index = start_index + offset

        items.append(
            {
                "id": (f"{book_name}_" f"{job_id}_" f"text_{index}"),
                "document": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    "page": chunk["page"],
                    "source": book_name,
                    "job_id": job_id,
                    "file_hash": file_hash,
                    "type": "text",
                },
            }
        )

    return items


def _embed_image_batch(batch, model, book_name, job_id, file_hash, start_index):

    texts = [chunk["text"] for chunk in batch]

    embeddings = model.encode(texts, batch_size=embedding_batch_size).tolist()

    items = []

    for offset, (chunk, embedding) in enumerate(zip(batch, embeddings)):

        index = start_index + offset

        items.append(
            {
                "id": (f"{book_name}_" f"{job_id}_" f"image_{index}"),
                "document": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    "page": chunk["page"],
                    "source": book_name,
                    "job_id": job_id,
                    "file_hash": file_hash,
                    "type": "image",
                },
            }
        )

    return items
