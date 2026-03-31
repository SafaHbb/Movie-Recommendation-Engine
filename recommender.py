import math
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DIMENSIONS = [
    "emotion_family",
    "emotion_romance",
    "adventure",
    "mystery",
    "philosophy",
    "worldbuilding",
    "humor",
    "darkness",
    "action",
    "character_depth",
    "visual_atmosphere",
    "pacing",
]

# Words associated with each dimension — used to build a user query text
KEYWORD_MAP = {
    "emotion_family": ["family", "father", "mother", "daughter", "son", "parent", "child", "home", "love"],
    "emotion_romance": ["romance", "relationship", "love story", "couple", "marriage", "heartbreak"],
    "adventure": ["journey", "quest", "adventure", "exploration", "travel", "expedition", "space travel"],
    "mystery": ["mystery", "secret", "investigation", "hidden", "discovery", "unknown"],
    "philosophy": ["philosophy", "meaning", "existential", "time", "destiny", "consciousness", "human nature"],
    "worldbuilding": ["fantasy", "universe", "world", "kingdom", "civilization", "future", "dystopia", "myth"],
    "humor": ["comedy", "funny", "humor", "satire", "witty"],
    "darkness": ["dark", "grief", "death", "trauma", "violence", "bleak", "sadness"],
    "action": ["battle", "war", "fight", "action", "chase", "explosion", "combat"],
    "character_depth": ["character", "growth", "transformation", "inner conflict", "psychological"],
    "visual_atmosphere": ["beautiful", "cinematic", "visual", "stylized", "atmospheric", "dreamlike"],
    "pacing": ["slow burn", "fast paced", "intense", "meditative", "suspense"],
}


def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required_columns = ["title", "overview", "genres", "tagline", "vote_average", "release_date"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    for col in ["title", "overview", "genres", "tagline"]:
        df[col] = df[col].fillna("").astype(str)

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["release_date"] = df["release_date"].fillna("").astype(str)

    df = df[df["title"].str.strip() != ""].copy()

    df["keywords"] = (df["tagline"] + " " + df["overview"]).str.strip()

    # Combined text used for embedding — richer = better
    df["combined_text"] = (
        df["title"] + ". " + df["overview"] + " " + df["genres"] + " " + df["keywords"]
    ).str.strip()

    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    return df


def fuzzy_find_movie(df: pd.DataFrame, title: str):
    choices = df["title"].dropna().unique().tolist()
    match = process.extractOne(title, choices, scorer=fuzz.WRatio)
    if not match:
        return None
    best_title, score, _ = match
    row = df[df["title"] == best_title].iloc[0]
    return row, score


def load_embedding_model():
    """Load the sentence-transformer model. Small and fast, runs locally."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_embedding_index(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """
    Embed all movie combined_text fields.
    Returns a 2D numpy array of shape (n_movies, embedding_dim).
    This is cached in Streamlit so it only runs once.
    """
    texts = df["combined_text"].fillna("").tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    return embeddings


def build_user_query(user_profile: dict, seed_movie_text: str) -> str:
    """
    Combine the seed movie text with a preference query built from yes/no answers.

    Scoring convention:
      5 = Yes (liked this aspect)
      3 = Not sure (neutral)
      1 = No (did not like / don't want this)

    Yes answers contribute their keywords to the query (stronger signal).
    No answers are not added — their absence already reduces similarity.
    The seed movie text anchors the overall vibe.
    """
    preference_words = []

    for dim, score in user_profile.items():
        words = KEYWORD_MAP.get(dim, [])
        if score >= 4:
            # Repeat keywords to give them more weight in the embedding
            preference_words.extend(words * 2)
        # score == 3 → neutral, skip
        # score == 1 → disliked, skip (absence reduces cosine similarity naturally)

    query = seed_movie_text + " " + " ".join(preference_words)
    return query.strip()


def rank_recommendations(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    user_profile: dict,
    seed_title: str,
    seed_movie_text: str,
    era: str = "Both",
    top_k: int = 5,
) -> list:
    """
    Rank movies using embedding cosine similarity.

    Strategy:
      - Build a query from seed movie text + user yes/no preferences
      - Embed the query
      - Score all movies by cosine similarity to that query
      - Apply negative penalty for dimensions the user said No to
      - Apply era filter
      - Add small rating boost
      - Return top_k results
    """
    # Build and embed the user query
    query_text = build_user_query(user_profile, seed_movie_text)
    query_embedding = model.encode([query_text])  # shape (1, dim)

    # Cosine similarity against all movies
    sims = cosine_similarity(query_embedding, embeddings).flatten()  # shape (n_movies,)

    # Apply negative penalty for dimensions user said No to
    # We check if each movie's text contains keywords from disliked dimensions
    disliked_dims = [dim for dim, score in user_profile.items() if score <= 1]

    results = []
    for i, row in df.iterrows():
        if row["title"].lower() == seed_title.lower():
            continue

        score = float(sims[i])

        # Penalise movies strong in dimensions the user didn't want
        if disliked_dims:
            movie_text_lower = row["combined_text"].lower()
            for dim in disliked_dims:
                dim_words = KEYWORD_MAP.get(dim, [])
                hit_count = sum(movie_text_lower.count(w.lower()) for w in dim_words)
                if hit_count > 2:
                    score -= 0.05 * min(hit_count, 5)  # cap penalty

        # Rating boost (small — 0 to 0.1)
        rating = float(row.get("vote_average", 0.0))
        score += rating / 100.0

        results.append({
            "title": row["title"],
            "overview": row["overview"],
            "genres": row["genres"],
            "score": score,
            "vote_average": rating,
            "release_date": row.get("release_date", ""),
        })

    # Era filter
    def parse_year(date_str):
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None

    if era == "New (2010+)":
        results = [r for r in results if (parse_year(r["release_date"]) or 0) >= 2010]
    elif era == "Classic (before 2000)":
        results = [r for r in results if (parse_year(r["release_date"]) or 9999) < 2000]

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]