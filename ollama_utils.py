import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_json(text: str):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start:end + 1])


def get_movie_aspects(title: str, overview: str, genres: str, keywords: str):
    prompt = f"""
You are helping build a movie recommendation app.

Movie title: {title}
Overview: {overview}
Genres: {genres}
Extra text: {keywords}

Task:
Identify 4 to 6 specific things a viewer might have enjoyed about this movie.
For each one, write a short yes/no question a friend might ask after watching the movie together.

Return ONLY valid JSON in this exact format:

{{
  "movie_title": "string",
  "aspects": [
    {{
      "aspect": "string",
      "dimension": "string",
      "question": "string"
    }}
  ]
}}

Rules for questions:
- Must be yes/no questions only. No scales, no open answers.
- Keep them casual and short — like a friend asking.
- Start with "Did you like..." or "Were you into..." or "Did you enjoy..."
- One idea per question. No "and" joining two things.

Good examples:
  "Did you like the romance in it?"
  "Were you into the action scenes?"
  "Did you like how dark and tense it got?"
  "Did you enjoy the humour?"
  "Were you into the family storyline?"
  "Did you like how it was set in space?"

Bad examples (do NOT do these):
  "Did you appreciate the nuanced character development and moral ambiguity?" (too complex)
  "Rate the pacing from 1 to 5." (not yes/no)
  "What did you think of the visuals?" (open ended)

The "dimension" must be one of:
  ["emotion_family", "emotion_romance", "adventure", "mystery", "philosophy",
   "worldbuilding", "humor", "darkness", "action", "character_depth",
   "visual_atmosphere", "pacing"]

Only include aspects clearly present in this movie.
Return JSON only. No explanation, no markdown.
"""
    raw = call_ollama(prompt)
    return extract_json(raw)


def explain_recommendations(seed_movie: str, user_profile: dict, recommendations: list):
    # Build readable taste summary from user answers
    liked = [dim for dim, score in user_profile.items() if score >= 4]
    disliked = [dim for dim, score in user_profile.items() if score <= 1]

    taste_summary = ""
    if liked:
        taste_summary += f"They liked: {', '.join(liked)}. "
    if disliked:
        taste_summary += f"They did NOT want: {', '.join(disliked)}."

    formatted_recs = "\n".join(
        [
            f"- {r['title']} | score={round(r['score'], 3)} | overview={r['overview'][:220]}"
            for r in recommendations[:5]
        ]
    )

    prompt = f"""
You are a friendly movie recommendation assistant.

The user loved this movie: {seed_movie}

Their taste: {taste_summary}

Top recommended movies:
{formatted_recs}

For each of the 5 movies, write 2-3 sentences explaining:
1. What specific thing makes it similar to {seed_movie}
2. Which part of the user's taste it matches

Keep it casual and short. Write it like a friend recommending a film.
Return plain text only.
"""
    return call_ollama(prompt)