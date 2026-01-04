import faiss
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# -----------------------------
# EMBEDDINGS (LangChain)
# -----------------------------

def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

# -----------------------------
# BUILD FAISS INDEX
# -----------------------------

def build_faiss(texts):
    embedding_model = get_embedding_model()

    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index

# -----------------------------
# SAVE FAISS INDEX
# -----------------------------

def save_faiss(index, texts):
    faiss.write_index(index, "data/faiss/index.faiss")
    with open("data/faiss/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

# -----------------------------
# SEARCH FAISS
# -----------------------------

def search_faiss(query, k=3):
    index = faiss.read_index("data/faiss/index.faiss")

    with open("data/faiss/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    _, indices = index.search(query_embedding, k)

    return [texts[i] for i in indices[0] if i != -1]
