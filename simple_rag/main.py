from fastapi import FastAPI, UploadFile, File
from vector_rag.pdf_loader import load_pdf_text as load_vector_pdf
from vector_rag.chunker import chunk_text
from vector_rag.vector_store import build_faiss, save_faiss, search_faiss

from kg_rag.pdf_loader import load_pdf_text as load_kg_pdf
from kg_rag.kg_builder import extract_entities_and_relations
from kg_rag.neo4j_store import Neo4jStore

from langchain_openai import ChatOpenAI
import os

app = FastAPI()

# ENV VARIABLES
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_API_KEY
)

neo4j_store = Neo4jStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# ----------------------------
# VECTOR RAG ENDPOINTS
# ----------------------------

@app.post("/upload/vector")
async def upload_vector_pdf(file: UploadFile = File(...)):
    text = load_vector_pdf(file.file)
    chunks = chunk_text(text)
    index = build_faiss(chunks, OPENAI_API_KEY)
    save_faiss(index, chunks)
    return {"status": "Vector DB created", "chunks": len(chunks)}

@app.post("/query/vector")
async def query_vector(query: str):
    docs = search_faiss(query, OPENAI_API_KEY)
    context = "\n".join(docs)

    response = llm.invoke(
        f"Answer using only this context:\n{context}\n\nQuestion: {query}"
    )

    return {"answer": response.content}

# ----------------------------
# KG RAG ENDPOINTS
# ----------------------------

@app.post("/upload/kg")
async def upload_kg_pdf(file: UploadFile = File(...)):
    text = load_kg_pdf(file.file)
    triples = extract_entities_and_relations(text)

    neo4j_store.clear()
    for e1, r, e2 in triples:
        neo4j_store.add_relation(e1, r, e2)

    return {"status": "Knowledge Graph created", "triples": len(triples)}

@app.get("/query/kg")
async def query_kg(entity: str):
    results = neo4j_store.query_entity(entity)
    return {
        "entity": entity,
        "relations": [
            f"{r['source']} -[{r['relation']}]-> {r['target']}"
            for r in results
        ]
    }
