import os
from dotenv import load_dotenv
from anthropic import Anthropic

SYSTEM_MESSAGE = (
    "You are a chatbot. "
    "You will have a conversation with a user. "
    "Be friendly and concise."
)


def main():
    # Load environment variables from .env
    load_dotenv()

    # Get API configuration
    API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    MODEL = os.environ.get("MODEL")

    # Validate environment variables
    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY was not found in the .env file")
        return

    if not MODEL:
        print("Error: MODEL was not found in the .env file")
        return

    # Create Anthropic client
    client = Anthropic(api_key=API_KEY)

    print(f"Chatting with model: {MODEL}")
    print("Type 'exit' to end the conversation.\n")

    conversation_history = []

    while True:
        user_message = input("> ")
        if user_message.lower() in ["exit", "quit"]:
            print("Conversation ended.")
            break
        conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_MESSAGE,
                messages=conversation_history
            )
            assistant_reply = response.content[0].text
            print(f"\nAssistant: {assistant_reply}\n")
            conversation_history.append({
                "role": "assistant",
                "content": assistant_reply
            })

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()