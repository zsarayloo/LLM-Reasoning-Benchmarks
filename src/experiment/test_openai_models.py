import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
    print("=== test_openai_models.py ===")
    print("Current working directory:", os.getcwd())

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY loaded:", "YES" if api_key else "NO")

    if not api_key:
        print("No API key found in environment. Please set OPENAI_API_KEY in your .env.")
        return

    client = OpenAI()

    question = "Solve 7 + 11 and answer only with the final number."

    models_to_test = [
        "gpt-4.1-mini",  # cheap sanity model
        "gpt-4.1",
        "gpt-5.1-mini",
        "gpt-5.1",
    ]

    for m in models_to_test:
        print(f"\n--- Testing model: {m} ---")
        try:
            response = client.responses.create(
                model=m,
                input=question,
            )
            text = response.output[0].content[0].text
            print("Answer:", text)
        except Exception as e:
            print("Error calling model:", repr(e))

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
