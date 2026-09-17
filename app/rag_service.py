from query_processor import route_question, decompose_question
from retriever import retrieve_chunks
from reranker import rerank_chunks
from generator import generate_answer
from config import retrieval_results, llm_context_chunks


def ask_question(question, search_all, selected_book, context):

    if route_question(question) == "COMPLEX":

        sub_questions = decompose_question(question)

        all_chunks = []
        merged_ranked = {}

        for sub_q in sub_questions:

            combined_chunks, ranked_chunks = retrieve_chunks(
                sub_q,
                search_all,
                selected_book,
                context.text_collection,
                context.ids,
                context.documents,
                context.metadatas,
                context.model,
            )

            all_chunks.extend(combined_chunks)
            merged_ranked.update(ranked_chunks)

        seen = set()
        combined_chunks = []

        for chunk in all_chunks:
            if chunk not in seen:
                seen.add(chunk)
                combined_chunks.append(chunk)

        ranked_chunks = merged_ranked

    else:

        combined_chunks, ranked_chunks = retrieve_chunks(
            question,
            search_all,
            selected_book,
            context.text_collection,
            context.ids,
            context.documents,
            context.metadatas,
            context.model,
        )

    combined_chunks = rerank_chunks(
        question, combined_chunks, ranked_chunks, context.cross_encoder
    )

    # Keep the top reranked retrieval results
    combined_chunks = combined_chunks[:retrieval_results]

    # Send only the best chunks to the LLM
    context_chunks = combined_chunks[:llm_context_chunks]

    context_text = "\n\n".join(context_chunks)

    answer = generate_answer(question, context_text)

    return answer
