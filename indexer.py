import pdfplumber
import os
import time
from utils import get_file_hash
from config import embedding_batch_size, book_folder
from preprocessor import clean_text
from vision import describe_image
import io
from semantic_chunker import semantic_chunk
from config import semantic_chunk_threshold, min_chunk_characters, max_chunk_characters


def index_books(text_collection, image_collection, model, splitter):
    book_files = os.listdir(book_folder)

    print(book_files)

    for book_name in book_files:
        if not book_name.endswith(".pdf"):
            continue

        pdf_path = os.path.join(book_folder, book_name)
        file_hash = get_file_hash(pdf_path)
        existing_book = text_collection.get(where={"source": book_name})
        should_index = False
        stored_hash = existing_book["metadatas"][0]["file_hash"] if existing_book["ids"] else None

        print(stored_hash)

        if existing_book["ids"]:
            if stored_hash == file_hash:
                print("Book unchanged, skipping indexing")
            else:
                print("Book has changed. Updating index...")
                text_collection.delete(where={"source": book_name})
                image_collection.delete(where={"source": book_name})
                should_index = True
        else:
            should_index = True

        if should_index:
            indexing_start = time.perf_counter()
            chunks = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = clean_text(page.extract_text())

                    if page_text:
                        page_chunks = semantic_chunk(
                            page_text,
                            model,
                            semantic_chunk_threshold,
                            min_chunk_characters,
                            max_chunk_characters
                        )
                        for chunk in page_chunks:
                            chunks.append({
                                "page": page_number,
                                "text": chunk,
                                "type": "text"
                            })

                    for img_data in page.images:
                        img_bytes = img_data["stream"].get_data()
                        img_path = f"temp_img_{book_name}_{page_number}.png"

                        with open(img_path, "wb") as img_file:
                            img_file.write(img_bytes)

                        description = describe_image(img_path)
                        chunks.append({
                            "page": page_number,
                            "text": f"[Image description: {description}]",
                            "type": "image"
                        })

                        os.remove(img_path)

            text_data = []
            image_data = []

            for chunk_data in chunks:
                if chunk_data.get("type") == "image":
                    image_data.append(chunk_data)
                else:
                    text_data.append(chunk_data)

            if text_data:
                text_ids = []
                text_documents = []
                text_embeddings = []
                text_metadatas = []

                for batch_start in range(0, len(text_data), embedding_batch_size):
                    batch = text_data[batch_start:batch_start + embedding_batch_size]
                    batch_texts = [chunk["text"] for chunk in batch]
                    batch_embeddings = model.encode(batch_texts, batch_size=embedding_batch_size).tolist()

                    for offset, (chunk_data, embedding) in enumerate(zip(batch, batch_embeddings)):
                        overall_index = batch_start + offset
                        text_ids.append(f"{book_name}_text_{overall_index}")
                        text_documents.append(chunk_data["text"])
                        text_embeddings.append(embedding)
                        text_metadatas.append({
                            "page": chunk_data["page"],
                            "source": book_name,
                            "file_hash": file_hash,
                            "type": "text"
                        })

                        if (overall_index + 1) % 50 == 0 or (overall_index + 1) == len(text_data):
                            print(f"> Indexed {overall_index + 1}/{len(text_data)} text chunks")

                text_collection.add(
                    ids=text_ids,
                    documents=text_documents,
                    embeddings=text_embeddings,
                    metadatas=text_metadatas
                )

            if image_data:
                image_ids = []
                image_documents = []
                image_embeddings = []
                image_metadatas = []

                for batch_start in range(0, len(image_data), embedding_batch_size):
                    batch = image_data[batch_start:batch_start + embedding_batch_size]
                    batch_texts = [chunk["text"] for chunk in batch]
                    batch_embeddings = model.encode(batch_texts, batch_size=embedding_batch_size).tolist()

                    for offset, (chunk_data, embedding) in enumerate(zip(batch, batch_embeddings)):
                        overall_index = batch_start + offset
                        image_ids.append(f"{book_name}_image_{overall_index}")
                        image_documents.append(chunk_data["text"])
                        image_embeddings.append(embedding)
                        image_metadatas.append({
                            "page": chunk_data["page"],
                            "source": book_name,
                            "file_hash": file_hash,
                            "type": "image"
                        })

                        if (overall_index + 1) % 50 == 0 or (overall_index + 1) == len(image_data):
                            print(f"> Indexed {overall_index + 1}/{len(image_data)} image chunks")

                image_collection.add(
                    ids=image_ids,
                    documents=image_documents,
                    embeddings=image_embeddings,
                    metadatas=image_metadatas
                )

            print(f"Indexed {len(text_data)} text chunks and {len(image_data)} image chunks")

            indexing_end = time.perf_counter()
            indexing_time = indexing_end - indexing_start
            print(f"Indexing completed in {indexing_time:.2f} seconds.")