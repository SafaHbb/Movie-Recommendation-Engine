"""
Run this ONCE before starting the app:

    python precompute_embeddings.py

Saves:
    data/movies_filtered.parquet
    data/embeddings.npy
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/movies_metadata.csv"
OUT_MOVIES = "data/movies_filtered.parquet"
OUT_EMBEDDINGS = "data/embeddings.npy"

MIN_VOTE_COUNT = 500
MIN_VOTE_AVERAGE = 6.0
MIN_OVERVIEW_LEN = 50


def load_and_filter(path: str) -> pd.DataFrame:
    print("Loading CSV...")
    df = pd.read_csv(path, low_memory=False)

    required = ["title", "overview", "genres", "tagline",
                "vote_average", "vote_count", "release_date", "poster_path"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    for col in ["title", "overview", "genres", "tagline", "poster_path"]:
        df[col] = df[col].fillna("").astype(str)

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0.0)
    df["release_date"] = df["release_date"].fillna("").astype(str)

    before = len(df)
    df = df[df["title"].str.strip() != ""]
    df = df[df["overview"].str.len() >= MIN_OVERVIEW_LEN]
    df = df[df["vote_average"] >= MIN_VOTE_AVERAGE]
    df = df[df["vote_count"]   >= MIN_VOTE_COUNT]
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    df["keywords"] = (df["tagline"] + " " + df["overview"]).str.strip()
    df["combined_text"] = (
        df["title"] + ". " + df["overview"] + " " + df["genres"] + " " + df["keywords"]
    ).str.strip()

    print(f"  {before:,} → {len(df):,} movies after filtering")
    return df


def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["combined_text"].tolist()
    print(f"Embedding {len(texts):,} movies...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


if __name__ == "__main__":
    df = load_and_filter(DATA_PATH)
    df.to_parquet(OUT_MOVIES, index=False)
    print(f"Saved filtered movies → {OUT_MOVIES}")

    embeddings = build_embeddings(df)
    np.save(OUT_EMBEDDINGS, embeddings)
    print(f"Saved embeddings → {OUT_EMBEDDINGS}")
    print(f"Shape: {embeddings.shape}")
    print("\n✓ Done. Now run: streamlit run app.py")