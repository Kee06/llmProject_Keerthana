**Task 1:Groq AI Chatbot**
The goal of this project was to build a terminal-based AI chatbot using the Groq API and Python.

The chatbot should:
- Accept user input
- Generate AI responses
- Remember previous conversations

I used:
- Python
- Groq API
- Llama 3.3 model

Features
- Multi-turn conversation
- Adjustable temperature
- System prompts
- Conversation memory
  
The files uploaded includes:
- chat.py => basic chatbot with single response
- chat2.py => multiturn conversation feature included
- chat3.py => trial with system message
- chat4.py => all features including temperature and max_tokens
  
Challenges Faced

 1. Model 
    The original model was shown as unsupported so I had to change it. I changed it to
    `llama-3.3-70b-versatile`
 2. Temperature
    Understanding the use of temperature was difficult but after testing with different values for the same questions it became clear. 
 3. Syntax
     There were some confusions regarding the exact syntax of the code since we had to use Groq

Steps to run it:

1. Clone the repository
git clone <repository-link>

2. Go to project folder
cd llmProject_Keerthana

3. Create virtual environment
python -m venv venv

4. Activate virtual environment
venv\Scripts\activate

5. Install dependencies
pip install groq

6. Add Groq API key
Replace api_key="YOUR_API_KEY" with your real API key.

7. Run the chatbot
python chat.py

[Type exit to stop the chatbot]


**Task 2:RAG AI system**
The goal of this project was to build a Retrieval-Augmented Generation (RAG) system using Python, LangChain, ChromaDB, HuggingFace embeddings, and the Groq API.

The RAG system should:
-Load PDF documents
-Split documents into chunks
-Generate embeddings for semantic search
-Retrieve relevant document chunks
-Generate context-aware AI responses

How Retrieval Works:
   The RAG system works by combining document search with AI. First, the PDF is loaded and split into smaller chunks of text. These chunks are converted into embeddings using a Hugging Face model. The embeddings are stored in ChromaDB. When the user asks a question, the question is also converted into an embedding. The system compares it with the stored embeddings and retrieves the most relevant chunks from the PDF. These chunks are then given as context to the Groq model, which generates the final answer based only on the retrieved information. This helps the chatbot answer questions related to the uploaded document instead of using only general AI knowledge.

I used:
-Python
-LangChain
-ChromaDB
-HuggingFace Embeddings
-Groq API
-Llama 3.1 model

Features
-PDF document loading
-Text chunking
-Embedding generation
-ChromaDB vector storage
-Semantic similarity search
-Multi-document retrieval
-Context-aware question answering
-Conversation logging

The files uploaded includes:
-rag.py => complete RAG pipeline implementation
-notes.pdf => input document for retrieval
-manual.pdf => additional document source
-research.pdf => additional document source
-requirements.txt => required libraries and dependencies

Challenges Faced

1. Import errors happened because newer versions of LangChain changed many module names and the .env file also caused an encoding error and had to be recreated in UTF-8 format.

2. There was repeated retrieval results. The system kept retrieving the same chunk multiple times, especially for short PDF pages. This problem was improved by removing duplicate chunks and using a better retrieval method called max_marginal_relevance_search().

3. There was also a model issue because the old Groq model used in the assignment had been removed, so it was replaced with a newer supported model.
4. Chat history memory was explored as an additional feature, but compatibility issues occurred due to newer LangChain version changes.

Possible improvements:

1. Use stronger embedding models for higher accuracy such as BGE embeddings and instructor embeddings
2. Combine semantic search and keyword search for better retrieval.
3. Instead of fixed-size chunks, use smarter chunking like semantic Chunking which splits based on meaning/topic changes instead of character count.
4. The system can automatically improve user questions to improve retrieval accuracy.

Steps to run it:

1. Clone the repository
   git clone <repository-link>

2. Go to project folder
   cd llmProject_Keerthana

3. Create virtual environment
   python -m venv venv

4. Activate virtual environment
   venv\Scripts\activate

5. Install dependencies
   pip install -r requirements.txt

6. Add Groq API key
   Create a `.env` file and add:
   GROQ_API_KEY="YOUR_API_KEY"

7. Run the RAG system
   python rag.py

[Type exit to stop the chatbot]


