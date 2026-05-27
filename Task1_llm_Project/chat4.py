from groq import Groq
from colorama import Fore, Style, init

init()

client = Groq(
    api_key="YOUR_API_KEY"
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
]

while True:
    user_input = input(Fore.CYAN + "You: " + Style.RESET_ALL)

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=200
    )

    reply = chat_completion.choices[0].message.content

    print(Fore.GREEN + "\nAI: " + reply + "\n" + Style.RESET_ALL)

    messages.append({
        "role": "assistant",
        "content": reply
    })