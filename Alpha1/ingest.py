import chromadb
import llama_index
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# establish connection with db
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("doc")

# setup embedder
model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# loading docs
docs = SimpleDirectoryReader("data").load_data()

vector_store= ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=model
)
print("data inserted")