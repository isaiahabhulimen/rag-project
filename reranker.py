from models import model, cross_encoder
from utils import diversify_chunks



def rerank_chunks(question, combined_chunks, ranked_chunks, cross_encoder):

    print("\n===== BEFORE DIVERSIFICATION =====")
    for chunk in combined_chunks[:10]:
        print(chunk[:120])
        print("----------------")
     #retrive top five
    #combined_chunks = diversify_chunks(combined_chunks, ranked_chunks, model, top_k=5)
    print("\n===== AFTER DIVERSIFICATION =====")
    for chunk in combined_chunks[:10]:
        print(chunk[:120])
        print("----------------")
   
    
    cross_encoder_inputs = []
    for chunk in combined_chunks:
        cross_encoder_inputs.append([question, chunk])
    cross_encoder_scores = cross_encoder.predict(cross_encoder_inputs)

    chunk_score_pairs = list(zip(combined_chunks, cross_encoder_scores))

    chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

    largest_gap = 0
    cutoff_index = len(chunk_score_pairs)

    for i in range(len(chunk_score_pairs) - 1):
        current_score = chunk_score_pairs[i][1]
        next_score = chunk_score_pairs[i + 1][1]
        gap = current_score - next_score

        if (
            gap > largest_gap
            and current_score > 0
            and gap > current_score * 0.25
        ):
            largest_gap = gap
            cutoff_index = i + 1

    minimum_chunks = 5

    cutoff_index = max(cutoff_index, minimum_chunks)

    chunk_score_pairs = chunk_score_pairs[:cutoff_index]



    print("\n===== AFTER CROSS ENCODER =====")
    for chunk, score in chunk_score_pairs:
        print("Score:", score)
        print(chunk[:120])
        print("----------------")

    

    combined_chunks = [chunk for chunk, score in chunk_score_pairs]
    
    return combined_chunks
