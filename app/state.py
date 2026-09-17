import chromadb

from models import model, cross_encoder
from config import collection_name
from app.context import AppContext

client = chromadb.PersistentClient(path="database")


text_collection = client.get_or_create_collection(name=f"{collection_name}_text")

image_collection = client.get_or_create_collection(name=f"{collection_name}_images")


all_documents = text_collection.get()

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
