Groq AI Chatbot
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
