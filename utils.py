import hashlib
from sentence_transformers.util import cos_sim


def get_file_hash(pdf_path):
    with open(pdf_path, "rb") as file:
        file_bytes = file.read()

    hash_object = hashlib.sha256()
    hash_object.update(file_bytes)

    file_hash = hash_object.hexdigest()

    return file_hash


def diversify_chunks(ranked_chunks, ranked_scores, model, top_k=5, lambda_value=0.7):
    if len(ranked_chunks) <= top_k:
        return ranked_chunks

    chunk_embeddings = model.encode(ranked_chunks, convert_to_tensor=True)
    selected_chunks = [ranked_chunks[0]]
    selected_indices = [0]

    remaining_indices = list(range(1, len(ranked_chunks)))

    while len(selected_chunks) < top_k and remaining_indices:

        best_candidate = None
        best_score = float("-inf")

        for candidate_index in remaining_indices:
            relevance = ranked_scores.get(ranked_chunks[candidate_index], 0)

            max_similarity = 0
            for selected_index in selected_indices:
                similarity = cos_sim(
                    chunk_embeddings[candidate_index], chunk_embeddings[selected_index]
                ).item()

                if similarity > max_similarity:
                    max_similarity = similarity

            mmr_score = (lambda_value * relevance) - (
                (1 - lambda_value) * max_similarity
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate_index

        selected_chunks.append(ranked_chunks[best_candidate])

        selected_indices.append(best_candidate)

        remaining_indices.remove(best_candidate)
    return selected_chunks
