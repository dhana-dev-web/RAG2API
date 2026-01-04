import streamlit as st
from vector_rag.pdf_loader import load_pdf_text as load_vector_pdf
from vector_rag.chunker import chunk_text
from vector_rag.vector_store import build_faiss, save_faiss, search_faiss

from kg_rag.pdf_loader import load_pdf_text
from kg_rag.kg_builder import extract_entities_and_relations
from kg_rag.neo4j_store import Neo4jStore
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "sm8fnLrcRmBKUhoABEjW2Zhaby5Wy43YksSiJ1xAvBM"

store = Neo4jStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

st.title("📄 Simple RAG Chatbot")

mode = st.radio("Choose Mode", ["Vector RAG", "Knowledge Graph RAG"])

if mode == "Knowledge Graph RAG": st.title("🧠 Knowledge Graph (Neo4j)")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
query = ""
query_entity = ""

if uploaded_pdf:
    if mode == "Vector RAG":
        text = load_vector_pdf(uploaded_pdf)
        chunks = chunk_text(text)
        index = build_faiss(chunks)
        save_faiss(index, chunks)
        st.success("Vector DB created")

    else:
        text = load_pdf_text(uploaded_pdf)
        triples = extract_entities_and_relations(text)

        store.clear()
        for e1, r, e2 in triples:
            store.add_relation(e1, r, e2)

        st.success("Knowledge Graph created in Neo4j")
        st.write("Number of entities (nodes):", store.get_entities())
        relations = store.get_relations()
        # for r in relations:
        #     st.write(f"{r['source']} -[{r['relation']}]-> {r['target']}")
        

    if mode == "Vector RAG": query = st.text_input("Ask a question")

    if mode == "Knowledge Graph RAG": query_entity = st.text_input("Query entity from KG")

context = ""
    
if query:
    if mode == "Vector RAG":
        docs = search_faiss(query)
        context = "\n".join(docs)

    else:
        if query_entity:
            results = store.query_entity(query_entity)
            if results:
                for r in results:
                    st.write(f"{r['source']} -[{r['relation']}]-> {r['target']}")
            else:
                st.warning("Entity not found")
    
    llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.7, 
                api_key=openai_api_key
            )


    if mode == "Vector RAG":
        answer_prompt = f"Answer using only this context:\n{context}\n\n"
        answer = llm.invoke([HumanMessage(content=answer_prompt)]).content
        st.write(answer)

    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     temperature=0,
    #     messages=[{
    #         "role": "user",
    #         "content": f"Answer using only this context:\n{context}\n\nQuestion:{query}"
    #     }],
    #     api_key=openai_api_key
    # )
   

