from ollama import Client

def main():
    model_name = "llama3.2:3b"  # IMPORTANT: matches `ollama list`
    client = Client()  # defaults to http://127.0.0.1:11434

    print("=== test_lama_models.py ===")
    print("Using model:", model_name)

    prompt = "Solve 3 + 5 and answer only with the number."

    print("Sending a test prompt to local model...")
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    print("\nRaw response object:")
    print(response)

    print("\nModel output content:")
    print(response["message"]["content"])

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
