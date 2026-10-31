# ingest.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import shutil
import os

def build_index(fresh: bool = False):
    if fresh and os.path.exists("./chroma_data"):
        print("Removing existing chroma_data ...")
        shutil.rmtree("./chroma_data")

    print("Loading PDFs from ./data ...")
    documents = SimpleDirectoryReader("data", recursive=True).load_data()
    print(f"Loaded {len(documents)} pages.")

    for doc in documents:
        path = doc.metadata["file_path"].lower()
        if "science" in path:
            doc.metadata["subject"] = "science"
        elif "english" in path:
            doc.metadata["subject"] = "english"
        else:
            doc.metadata["subject"] = "unknown"

    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    chroma_collection = chroma_client.get_or_create_collection(name="education")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(f"✅ Ingestion complete. Total vectors: {chroma_collection.count()}")

if __name__ == "__main__":
    # Change fresh=True to wipe and rebuild
    build_index(fresh=False)