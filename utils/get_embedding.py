from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "embeddings.json"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Ensure all keys are str
                return {str(k): v for k, v in data.items()}
            except json.JSONDecodeError:
                return {}
    else:
        return {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

def get_embedding(text):
    """
    This method will get the embedding of the text using the OpenAI API.
    If the embedding is saved in embeddings.json, it will use the cached value.
    Otherwise, it will compute, save, and return it.
    """
    text_key = str(text).strip()
    embeddings = load_embeddings()
    if text_key in embeddings:
        return embeddings[text_key]
    response = client.embeddings.create(input=text_key, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    embeddings[text_key] = embedding
    save_embeddings(embeddings)
    return embedding