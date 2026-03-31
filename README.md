# Movie Recommendation Engine

A local movie recommendation system powered by LLMs using Ollama.

## What it does
Takes a movie you like and recommends similar ones based on semantic similarity using embeddings.

## Tech Stack
- Python
- Ollama (local LLM)
- Sentence embeddings
- Streamlit (app.py)

## How to run

1. Install dependencies:
   pip install -r requirements.txt

2. Precompute embeddings:
   python precompute_embeddings.py

3. Run the app:
   streamlit run app.py

## Requirements
- Ollama installed and running locally
