from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import *
from models import model,cross_encoder
from utils import get_file_hash,diversify_chunks
from retriever import retrieve_chunks
from reranker import rerank_chunks
from generator import generate_answer

from benchmark_loader import load_benchmark
from evaluation import evaluate_rag
from query_processor import route_question, decompose_question

from app.context import AppContext
from app.rag_service import ask_question




# Vector database client 
client = chromadb.PersistentClient(path="database")


# Create the collection

text_collection = client.get_or_create_collection(name=f"{collection_name}_text")
image_collection = client.get_or_create_collection(name=f"{collection_name}_images")


#create the chunking tool
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


#fetch document (only from text collection)
all_documents = text_collection.get()
print(all_documents.keys())
ids = all_documents["ids"]
documents = all_documents["documents"]
metadatas = all_documents["metadatas"]

app_context = AppContext(
    text_collection=text_collection,
    image_collection=image_collection,
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    model=model,
    cross_encoder=cross_encoder,
)

#evaluation
print("\nSystem Initialized. Select Execution Mode:")
print("1. Interactive Chat Mode")
print("2. Benchmark Evaluation Mode")
mode = input("Enter 1 or 2:")
if mode == "2":
    print("\nStarting Evaluation mode...")
    
    benchmark = load_benchmark("sample_benchmark.json")
    report = evaluate_rag(benchmark, text_collection, ids, documents, metadatas, model, cross_encoder)
    print(report)
elif mode =="1":

    while True:

        search_all = input("\nSearch all books? (yes/no): ").lower()

        selected_book = None
        if search_all == "no":
            selected_book = input("Enter the book name: ")

        question = input("\nAsk a question (or type 'exit' to quit): ")

        if question.lower()== "exit":

            print("\nGoodbye!")

            break      


        answer = ask_question(
            question=question,
            search_all=search_all,
            selected_book=selected_book,
            context=app_context
        )
        

        
        print()

        print("=================================================")
        print("RAG Answer")
        print("=================================================")

        print(answer)

        print("=================================================")

else:
    print("Invalid selection. Please restart and choose 1 or 2.")





