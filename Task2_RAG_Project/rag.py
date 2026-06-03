import os
import logging
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(
    filename="rag_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

pdf_files = [
    "notes.pdf",
    "manual.pdf",
    "research.pdf"
]

documents = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs = loader.load()

    # clean text
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    documents.extend(docs)

print(f"\nLoaded {len(documents)} pages\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks\n")

print("\n===== SAMPLE CHUNKS =====\n")

for i, chunk in enumerate(chunks[:2]):
    print(f"\nChunk {i+1}:\n")
    print(f"Source: {chunk.metadata.get('source')}")
    print(chunk.page_content)
    print("\n" + "=" * 50)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\nStored in ChromaDB\n")

llm = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY")
    model="llama-3.1-8b-instant"
)
prompt_template = """
Answer ONLY using the context below.

If the answer is not in the context, say "I don't know based on the given documents."

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

while True:

    question = input("\nAsk a question (type exit to quit): ")

    if question.lower() == "exit":
        break

    logging.info(f"Question: {question}")

    retrieved_docs = vector_db.max_marginal_relevance_search(
        question,
        k=3,
        fetch_k=10
    )

    print("\n===== RETRIEVED CHUNKS =====\n")

    context = ""

    for i, doc in enumerate(retrieved_docs):

        print(f"\nChunk {i+1}:\n")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content)
        print("\n" + "=" * 50)

        context += f"[Page {doc.metadata.get('page')}]\n{doc.page_content}\n\n"

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    logging.info(f"Answer: {response.content}")

    print("\n===== FINAL ANSWER =====\n")
    print(response.content)

    print("\n===== SOURCES =====\n")

    for doc in retrieved_docs:
        print(f"Source PDF: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")

    print("\n" + "#" * 60)