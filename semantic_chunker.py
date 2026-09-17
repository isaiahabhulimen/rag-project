import re
import numpy as np


# Step 2: Split into sentences
def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# Step 12: Update embedding (average)
def update_embedding(current_embedding, new_embedding, count):
    return (current_embedding * count + new_embedding) / (count + 1)


# Steps 3-14: Main chunking function
def semantic_chunk(text, model, similarity_threshold, min_chunk_size, max_chunk_size):
    # Step 4: Get sentences
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # Step 5: Handle single sentence
    if len(sentences) == 1:
        return [sentences[0]]

    # Step 6: Get embeddings
    # Use NumPy instead of PyTorch tensors to reduce temporary memory usage.
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    # Step 7: Initialize
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = sentence_embeddings[0]

    # Step 8: Loop through remaining sentences
    for i in range(1, len(sentences)):

        sentence = sentences[i]
        sentence_embedding = sentence_embeddings[i]

        # Step 9: Calculate cosine similarity
        similarity = np.dot(current_embedding, sentence_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(sentence_embedding)
        )

        # Step 10: Create candidate
        candidate_chunk = current_chunk + [sentence]
        candidate_text = " ".join(candidate_chunk)

        # Step 11: Decision
        if similarity >= similarity_threshold and len(candidate_text) <= max_chunk_size:
            # Keep together
            current_chunk = candidate_chunk

            current_embedding = update_embedding(
                current_embedding, sentence_embedding, len(current_chunk) - 1
            )

        else:
            # Split
            if len(" ".join(current_chunk)) >= min_chunk_size:

                chunks.append(" ".join(current_chunk))

                current_chunk = [sentence]
                current_embedding = sentence_embedding

            else:
                # Force keep if too small
                current_chunk = candidate_chunk

                current_embedding = update_embedding(
                    current_embedding, sentence_embedding, len(current_chunk) - 1
                )

    # Step 13: Save last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Step 14: Release temporary sentence embeddings
    del sentence_embeddings

    # Return
    return chunks
