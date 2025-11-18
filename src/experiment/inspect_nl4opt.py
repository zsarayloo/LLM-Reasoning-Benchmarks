import os
import pandas as pd
from dotenv import load_dotenv


def main():
    print("=== Starting inspect_nl4opt.py ===")

    # 1) Load .env
    load_dotenv()
    hf_token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    print(f"HuggingFace token loaded: {'YES' if hf_token else 'NO'}")

    # 2) Define the path on Hugging Face
    path = "hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json"
    print(f"Reading dataset from: {path}")

    # 3) Try loading with pandas + storage_options
    try:
        df = pd.read_json(
            path,
            lines=True,
            storage_options={"token": hf_token} if hf_token else None,
        )
        print("Successfully loaded DataFrame.")
    except Exception as e:
        print("ERROR while loading with pandas.read_json:")
        print(repr(e))
        print("\nAs a fallback, trying with datasets.load_dataset...")
        from datasets import load_dataset

        ds = load_dataset("CardinalOperations/NL4OPT", split="train")
        df = ds.to_pandas()
        print("Successfully loaded via datasets and converted to pandas.")

    # 4) Show some basic info
    print("\nDataFrame shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # Try to guess interesting columns to display
    cols_to_show = [
        c for c in df.columns
        if any(k in c.lower() for k in ["question", "problem", "answer", "solution"])
    ]
    print("\nColumns chosen for preview:", cols_to_show[:5])

    print("\n=== First 3 rows ===")
    print(df[cols_to_show].head(3).to_string(index=False))

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
