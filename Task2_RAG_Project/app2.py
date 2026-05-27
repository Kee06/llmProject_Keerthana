import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# LOAD ENV VARIABLES
load_dotenv()


# PAGE TITLE
st.title("Multi-Turn RAG Chatbot")


# LOAD EVERYTHING ONLY ONCE
@st.cache_resource
def load_rag_system():

    # LOAD PDF
    loader = PyPDFLoader("notes.pdf")

    documents = loader.load()

    # SPLIT DOCUMENTS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # EMBEDDING MODEL
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # VECTOR DATABASE
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    # LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    return vector_db, llm


# LOAD SYSTEM
vector_db, llm = load_rag_system()


# PROMPT TEMPLATE
prompt_template = """
You are a helpful AI assistant.

Use the conversation history and context below
to answer the user's question.

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"]
)


# SESSION STATE FOR CHAT HISTORY
if "messages" not in st.session_state:

    st.session_state.messages = []


# DISPLAY OLD MESSAGES
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# CHAT INPUT
question = st.chat_input("Ask your question")


# PROCESS QUESTION
if question:

    # SHOW USER MESSAGE
    with st.chat_message("user"):

        st.markdown(question)

    # SAVE USER MESSAGE
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("Searching document..."):

        # RETRIEVE DOCUMENTS
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

        # CREATE CHAT HISTORY
        chat_history = ""

        for msg in st.session_state.messages:

            role = msg["role"]

            content = msg["content"]

            chat_history += f"{role}: {content}\n"

        # FINAL PROMPT
        final_prompt = prompt.format(
            chat_history=chat_history,
            context=context,
            question=question
        )

        # GENERATE RESPONSE
        response = llm.invoke(final_prompt)

        answer = response.content

    # SHOW AI RESPONSE
    with st.chat_message("assistant"):

        st.markdown(answer)

    # SAVE AI RESPONSE
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # OPTIONAL: SHOW RETRIEVED CHUNKS
    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(retrieved_docs):

            st.write(f"Chunk {i+1}")

            st.write(f"Page: {doc.metadata.get('page')}")

            st.write(doc.page_content)

            st.write("--------------------------------")