from ollama import Client


def main():
    # By default, this talks to http://127.0.0.1:11434
    client = Client()

    model_name = "llama3.2:3"  # you can change to "llama3.2:3b" or "llama3:latest" later

    print(f"Sending a test prompt to local model: {model_name} ...")

    response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Solve 3 + 5 and answer only with the number.",
            }
        ],
    )

    print("\nRaw response dict:")
    print(response)

    # Extract the text content
    content = response["message"]["content"]
    print("\nModel output content:")
    print(content)


if __name__ == "__main__":
    main()
