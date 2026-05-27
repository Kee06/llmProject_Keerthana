import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate


# LOAD ENV
load_dotenv()


# PAGE TITLE
st.title("RAG Chatbot")


# LOAD EVERYTHING ONLY ONCE
@st.cache_resource
def load_rag_system():

    # LOAD PDF
    loader = PyPDFLoader("notes.pdf")

    documents = loader.load()

    # SPLIT TEXT
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # EMBEDDINGS
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # CHROMADB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    # GROQ MODEL
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    return vector_db, llm


# LOAD SYSTEM
vector_db, llm = load_rag_system()


# PROMPT
prompt_template = """
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# INPUT BOX
question = st.text_input("Ask a question")


# QUESTION PROCESSING
if question:

    with st.spinner("Searching document..."):

        # RETRIEVE DOCS
        retrieved_docs = vector_db.max_marginal_relevance_search(
            question,
            k=5,
            fetch_k=10
        )

        # REMOVE DUPLICATES
        unique_docs = []

        seen = set()

        for doc in retrieved_docs:

            if doc.page_content not in seen:

                unique_docs.append(doc)

                seen.add(doc.page_content)

        retrieved_docs = unique_docs

        # CREATE CONTEXT
        context = ""

        for doc in retrieved_docs:

            context += doc.page_content + "\n\n"

        # FINAL PROMPT
        final_prompt = prompt.format(
            context=context,
            question=question
        )

        # GENERATE ANSWER
        response = llm.invoke(final_prompt)

        # SHOW ANSWER
        st.subheader("Answer")

        st.write(response.content)

        # SHOW CHUNKS
        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(retrieved_docs):

            st.write(f"Chunk {i+1}")

            st.write(f"Page: {doc.metadata.get('page')}")

            st.write(doc.page_content)

            st.write("--------------------------------")