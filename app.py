import streamlit as st

from ollama_utils import get_movie_aspects, explain_recommendations
from recommender import (
    load_movies,
    fuzzy_find_movie,
    load_embedding_model,
    build_embedding_index,
    rank_recommendations,
    DIMENSIONS,
)

st.set_page_config(page_title="Movie Taste Recommender", layout="wide")
st.title("🎬 Movie Recommendation Engine")
st.write("Tell me a movie you love. I'll ask what you liked about it, then find films that match your taste.")

DATA_PATH = "data/movies_metadata.csv"


@st.cache_resource
def load_model():
    """Load the embedding model once and keep it in memory."""
    return load_embedding_model()


@st.cache_data
def get_data():
    """Load movies and build the embedding index. Cached so it only runs once."""
    df = load_movies(DATA_PATH)
    return df


@st.cache_data
def get_embeddings(_df):
    """
    Build embeddings separately — decorated with _df to avoid hashing the dataframe.
    Uses st.cache_data so embeddings are computed once and reused.
    """
    model = load_model()
    embeddings = build_embedding_index(_df, model)
    return embeddings


# --- Load everything ---
with st.spinner("Loading model and data..."):
    model = load_model()
    df_movies = get_data()
    embeddings = get_embeddings(df_movies)

# --- Session state init ---
if "aspects_data" not in st.session_state:
    st.session_state.aspects_data = None

if "selected_movie_row" not in st.session_state:
    st.session_state.selected_movie_row = None

# --- Step 1: Movie input ---
st.markdown("---")
movie_input = st.text_input("Enter a movie you love", placeholder="e.g. Interstellar")

if st.button("Analyse movie & generate questions"):
    if not movie_input.strip():
        st.warning("Please enter a movie title.")
    else:
        result = fuzzy_find_movie(df_movies, movie_input)
        if result is None:
            st.error("Movie not found in dataset.")
        else:
            row, confidence = result
            st.session_state.selected_movie_row = row.to_dict()
            st.session_state.aspects_data = None  # reset on new search

            with st.spinner(f"Analysing '{row['title']}' with Ollama..."):
                aspects = get_movie_aspects(
                    title=row["title"],
                    overview=row["overview"],
                    genres=row["genres"],
                    keywords=row["keywords"],
                )
                st.session_state.aspects_data = aspects

            st.success(f"Matched to: **{row['title']}** (confidence: {confidence:.0f}%)")

# --- Step 2: Yes/No questions ---
if st.session_state.aspects_data and st.session_state.selected_movie_row:
    seed_row = st.session_state.selected_movie_row
    seed_title = seed_row["title"]

    st.markdown("---")
    st.subheader(f"What did you like about *{seed_title}*?")
    st.caption("Answer yes or no — or 'not sure' if you don't care either way.")

    aspects = st.session_state.aspects_data.get("aspects", [])
    answers = {}

    cols = st.columns(2)
    for i, aspect in enumerate(aspects):
        dim = aspect["dimension"]
        question = aspect["question"]

        with cols[i % 2]:
            answer = st.radio(
                question,
                options=["Yes", "Not sure", "No"],
                index=1,  # default to "Not sure"
                horizontal=True,
                key=f"aspect_{i}_{dim}",
            )

            if answer == "Yes":
                answers[dim] = 5
            elif answer == "No":
                answers[dim] = 1
            else:
                answers[dim] = 3

    # --- Era preference ---
    st.markdown("---")
    era_choice = st.radio(
        "Do you want recommendations to be newer or older films?",
        options=["Both", "New (2010+)", "Classic (before 2000)"],
        horizontal=True,
        index=0,
    )

    # --- Step 3: Get recommendations ---
    st.markdown("")
    if st.button("Get recommendations →"):
        user_profile = {dim: 3 for dim in DIMENSIONS}  # default neutral
        for dim, score in answers.items():
            user_profile[dim] = score

        seed_text = seed_row.get("combined_text", seed_row.get("overview", ""))

        with st.spinner("Finding your best matches..."):
            recommendations = rank_recommendations(
                df=df_movies,
                embeddings=embeddings,
                model=model,
                user_profile=user_profile,
                seed_title=seed_title,
                seed_movie_text=seed_text,
                era=era_choice,
                top_k=5,
            )

        if not recommendations:
            st.warning("No recommendations found for that era filter. Try selecting 'Both'.")
        else:
            st.markdown("---")
            st.subheader("Your top 5 recommendations")

            for i, rec in enumerate(recommendations, start=1):
                year = str(rec["release_date"])[:4] if rec["release_date"] else "Unknown"
                rating = f"⭐ {rec['vote_average']:.1f}" if rec["vote_average"] > 0 else ""

                with st.expander(f"{i}. {rec['title']} ({year})  {rating}", expanded=(i == 1)):
                    st.write(f"**Genres:** {rec['genres']}")
                    st.write(rec["overview"])
                    st.caption(f"Match score: {rec['score']:.4f}")

            # --- LLM explanations ---
            with st.spinner("Generating explanations with Ollama..."):
                explanation = explain_recommendations(
                    seed_movie=seed_title,
                    user_profile=user_profile,
                    recommendations=recommendations,
                )

            st.markdown("---")
            st.subheader("Why these films match your taste")
            st.write(explanation)