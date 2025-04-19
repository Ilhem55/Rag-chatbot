import os
import tempfile
from io import BytesIO
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Chat system prompt
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# Function to process the document and split it into chunks
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    file_data = uploaded_file.read()
    temp_file = BytesIO(file_data)

    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_pdf_file:
        temp_pdf_file.write(file_data)
        temp_pdf_filename = temp_pdf_file.name

    loader = PyMuPDFLoader(temp_pdf_filename)
    docs = loader.load()

    os.remove(temp_pdf_filename)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# Function to get or create the vector collection
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

# Function to add document chunks to the collection
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("✅ Données ajoutées à la base vectorielle !")

# Function to query the collection based on a prompt
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

# Function to call the LLM and generate a response
def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"}
        ]
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

# Function to re-rank documents using a cross-encoder
def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

# Page configuration
st.set_page_config(page_title="💬 Chat avec vos documents", page_icon="📄")
st.title("💬 Assistant IA Documentaire")
st.markdown("Posez une question en lien avec le document que vous avez uploadé.")

# Sidebar : chargement PDF
with st.sidebar:
    uploaded_file = st.file_uploader("📑 Upload PDF", type=["pdf"], accept_multiple_files=False)
    process = st.button("⚡️ Process")
    if uploaded_file and process:
        all_splits = process_document(uploaded_file)
        add_to_vector_collection(all_splits, uploaded_file.name)

# Initialisation de l'historique de conversation
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Affichage des messages précédents façon ChatGPT
for entry in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["response"])

# Entrée utilisateur façon chat
prompt = st.chat_input("Pose ta question ici...")

# Traitement de la question
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    results = query_collection(prompt)
    context = results.get("documents")[0]
    relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
    response = call_llm(context=relevant_text, prompt=prompt)

    # Affichage réponse IA avec spinner
    with st.chat_message("assistant"):
        with st.spinner("L'IA rédige sa réponse..."):
            response_text = ""
            for chunk in response:
                response_text += chunk
            st.markdown(response_text)

    # Sauvegarde dans l'historique
    st.session_state.conversation_history.append({
        "question": prompt,
        "response": response_text
    })
