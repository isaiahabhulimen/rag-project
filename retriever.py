from config import retrieval_results, phrase_bonus, max_frequency_bonus, semantic_weight, keyword_weight, print_image_results,debug_keyword_results,debug_semantic_results,retrieval_candidate_pool
from app.exceptions import RetrievalError
from rank_bm25 import BM25Okapi




def retrieve_chunks(question, search_all, selected_book, collection, ids, documents, metadatas, model):
    try:
        question_embedding = model.encode(question).tolist()   
 
    except Exception as e:
        raise RetrievalError("Failed to generate query embedding") from e

    
    question_words = question.lower().split()

    stop_words = {"the", "is", "a", "of", "in", "on", "at", "what", "who", "where", "when", "why", "how", "do", "does"}

    question_words = [word for word in question_words if word not in stop_words]
    try:
        tokenized_documents = [document.lower().split() for document in documents]

        bm25 = BM25Okapi(tokenized_documents)

        bm25_scores = bm25.get_scores(question_words)
    except Exception as e:
        raise RetrievalError("Keyword retrieval failed") from e

    top_bm25 = sorted(
    zip(bm25_scores, ids, documents, metadatas),
    key=lambda x: x[0],
    reverse=True)[:retrieval_candidate_pool]

    bm25_score_dict = {}
    chunk_metadata = {}
    chunk_text = {}
    for score, chunk_id, document, metadata in top_bm25:
        bm25_score_dict[chunk_id] = score
        chunk_metadata[chunk_id] = metadata
        chunk_text[chunk_id] = document


    print("Question:", question)
    print("Question words:", question_words)

    try:
        if search_all == "yes":

            results = collection.query(query_embeddings=[question_embedding], n_results=retrieval_candidate_pool, include=["documents", "metadatas", "distances"])  #results stores what chromadb returns,q=q uses the users question to check
        else:
            results = collection.query(query_embeddings=[question_embedding], n_results=retrieval_candidate_pool, where={"source": selected_book}, include=["documents", "metadatas", "distances"]) 
    except Exception as e:
        raise RetrievalError("Vector database retrieval failed") from e

    retrieved_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    semantic_distances = results["distances"][0]
    

    document_to_id = {}

    for chunk_id, document in zip(ids, documents):
        document_to_id[document] = chunk_id

    retrieved_ids = []

    for chunk, metadata in zip(retrieved_chunks, retrieved_metadata):
        chunk_id = document_to_id[chunk]
        retrieved_ids.append(chunk_id)
        chunk_metadata[chunk_id] = metadata

    semantic_scores = {}
    

    #create ranking dictionary/
    ranked_chunks = {}
    #semaantic results
    semantic_distances = results["distances"][0]
    print("\n==== Semantic Results ====")


    for chunk_id, chunk, metadata, distance in list(zip(retrieved_ids,retrieved_chunks,retrieved_metadata,semantic_distances))[:10]:

        chunk_type = metadata.get("type", "text")
        
        # Skip printing images if toggle is False
        if chunk_type == "image" and not print_image_results:
            continue
        
        print("Type:", chunk_type.upper())
        print("Distance:", distance)
        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print(chunk[:150])
        print("------------------------")

        
        semantic_scores[chunk_id] = 1 / (1 + distance)
        chunk_metadata[chunk_id] = metadata
        chunk_text[chunk_id] = chunk
        #semantic and keyword normalization blocks
#normalization
    if semantic_scores:
        highest_semantic = max(semantic_scores.values())

        lowest_semantic = min(semantic_scores.values())

        if highest_semantic == lowest_semantic:
            for chunk in semantic_scores:
                semantic_scores[chunk] = 1
        else:

            for chunk in semantic_scores:

                semantic_scores[chunk] = (semantic_scores[chunk] - lowest_semantic) / (highest_semantic - lowest_semantic)
    
    if bm25_score_dict:
        highest_bm25 = max(bm25_score_dict.values())
        lowest_bm25 = min(bm25_score_dict.values())

        if highest_bm25 == lowest_bm25:
            for chunk in bm25_score_dict:
                bm25_score_dict[chunk] = 1
        else:
            for chunk in bm25_score_dict:
                bm25_score_dict[chunk] = (bm25_score_dict[chunk] - lowest_bm25) / (highest_bm25 - lowest_bm25)

     #weighted fusion block

    ranked_chunks = {}
    for chunk in semantic_scores:
        ranked_chunks[chunk] = semantic_scores[chunk] * semantic_weight

    for chunk in bm25_score_dict: 
        if chunk in ranked_chunks: 
            ranked_chunks[chunk] += bm25_score_dict[chunk] * keyword_weight 
        else: ranked_chunks[chunk] = bm25_score_dict[chunk] * keyword_weight
    

    combined_chunks = sorted(ranked_chunks, key=ranked_chunks.get, reverse=True)

    print("\n==== Hybrid Ranking ====")

    for chunk_id in combined_chunks[:debug_semantic_results]:
        print("Score:", ranked_chunks[chunk_id])
        print("Source:", chunk_metadata[chunk_id]["source"])
        print(chunk_text[chunk_id][:150])
        print("---------------------")
       # Multi-document balancing using metadata
    """
    source_groups = {}
    for chunk_id in combined_chunks:
        if chunk_id not in chunk_metadata:
            print("Missing metadata:", chunk_id)
            continue

        source = chunk_metadata[chunk_id]["source"]
        
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(chunk_text[chunk_id])

    balanced_chunks = []
    max_per_source = max(1, retrieval_results // len(source_groups)) if source_groups else retrieval_results

    for source in source_groups:
        balanced_chunks.extend(source_groups[source][:max_per_source])

    combined_chunks = balanced_chunks[:retrieval_results]
    """

    combined_chunks = [chunk_text[chunk_id]
    for chunk_id in combined_chunks[:retrieval_results]]

    return combined_chunks, ranked_chunks
