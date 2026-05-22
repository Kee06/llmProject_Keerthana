from groq import Groq

client = Groq(
    api_key="YOUR_API_KEY"
)

messages = [ {
        "role": "system",
        "content": "You are a helpful assistant."
    }]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile", temperature=1.2
    )

    reply = chat_completion.choices[0].message.content

    print("\nAI:", reply, "\n")

    messages.append({
        "role": "assistant",
        "content": reply
    })