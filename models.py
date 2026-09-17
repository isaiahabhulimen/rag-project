from sentence_transformers import SentenceTransformer, CrossEncoder

from config import model_name, cross_encoder_name

# initialise dense embeddings
model = SentenceTransformer(model_name, device="cpu")

# initialise cross encoder for reranking
cross_encoder = CrossEncoder(cross_encoder_name)
