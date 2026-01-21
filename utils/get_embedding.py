from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Union, Optional

load_dotenv()

EMBEDDINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "embeddings.json"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global in-memory cache
_EMBEDDINGS_CACHE: Optional[Dict[str, List[float]]] = None

def load_embeddings() -> Dict[str, List[float]]:
    """Load embeddings from file into global cache if not already loaded."""
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Ensure all keys are str and values are lists
                    _EMBEDDINGS_CACHE = {str(k): v for k, v in data.items()}
                except json.JSONDecodeError:
                    _EMBEDDINGS_CACHE = {}
        else:
            _EMBEDDINGS_CACHE = {}
    return _EMBEDDINGS_CACHE

def save_embeddings(embeddings: Optional[Dict[str, List[float]]] = None):
    """Save embeddings to file. If embeddings is None, saves the global cache."""
    if embeddings is None:
        embeddings = _EMBEDDINGS_CACHE
    if embeddings is not None:
        with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)

def get_embedding(text: str) -> List[float]:
    """
    Get the embedding of the text using the OpenAI API.
    If the embedding is cached in memory, uses the cached value.
    Otherwise, computes, caches, and returns it.
    """
    text_key = str(text).strip()
    embeddings = load_embeddings()
    if text_key in embeddings:
        return embeddings[text_key]

    # Compute new embedding
    response = client.embeddings.create(input=text_key, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    embeddings[text_key] = embedding
    print(f"Computed embedding for: {text}")
    save_embeddings(embeddings)
    return embedding

def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> Dict[str, List[float]]:
    """
    Get embeddings for a list of texts in batches.
    Returns a dictionary mapping text keys to embeddings.
    Only computes embeddings for texts not already cached.
    """
    embeddings = load_embeddings()
    results = {}

    # Separate cached and uncached texts
    uncached_texts = []
    for text in texts:
        text_key = str(text).strip()
        if text_key in embeddings:
            results[text_key] = embeddings[text_key]
        else:
            uncached_texts.append(text_key)

    # Process uncached texts in batches
    for i in range(0, len(uncached_texts), batch_size):
        batch_texts = uncached_texts[i:i + batch_size]
        print(f"Computing embeddings for batch {i//batch_size + 1}: {len(batch_texts)} texts")

        try:
            response = client.embeddings.create(input=batch_texts, model="text-embedding-3-small")
            for text, embedding_data in zip(batch_texts, response.data):
                embeddings[text] = embedding_data.embedding
                results[text] = embedding_data.embedding
        except Exception as e:
            print(f"Error computing batch embeddings: {e}")
            # Fallback to individual requests for failed batch
            for text in batch_texts:
                if text not in results:
                    try:
                        results[text] = get_embedding(text)
                    except Exception as e2:
                        print(f"Failed to get embedding for '{text}': {e2}")
                        raise

    # Save updated embeddings
    save_embeddings(embeddings)
    return results

def preload_embeddings(texts: List[str], batch_size: int = 100) -> None:
    """Preload embeddings for a list of texts into the cache."""
    get_embeddings_batch(texts, batch_size)

def clear_cache() -> None:
    """Clear the in-memory embeddings cache."""
    global _EMBEDDINGS_CACHE
    _EMBEDDINGS_CACHE = None