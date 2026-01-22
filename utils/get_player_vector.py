from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

PLAYERS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "players.json"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global in-memory cache
_PLAYERS_CACHE: Optional[Dict[str, List[float]]] = None

def load_players() -> Dict[str, List[float]]:
    """Load player vectors from file into global cache if not already loaded."""
    global _PLAYERS_CACHE
    if _PLAYERS_CACHE is None:
        if os.path.exists(PLAYERS_PATH):
            with open(PLAYERS_PATH, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Ensure all keys are str and values are lists
                    _PLAYERS_CACHE = {str(k): v for k, v in data.items()}
                except json.JSONDecodeError:
                    _PLAYERS_CACHE = {}
        else:
            _PLAYERS_CACHE = {}
    return _PLAYERS_CACHE


def describe_trait_score(value: float) -> str:
    """
    Convert a numeric personality or psychological trait score
    into a more nuanced textual category.
    """
    if value <= 2.0:
        return "very low"
    elif value <= 2.5:
        return "low"
    elif value <= 3.5:
        return "moderate"
    elif value <= 4.0:
        return "high"
    else:
        return "very high"



def _player_features_to_string(player_features: Dict) -> str:
    """Convert player features dictionary to a standardized string representation."""
    features_string = f"""
        Marriage: {player_features['marriage']} 
        Education: {player_features['education']}
        Race: {player_features['race']}
        Continent: {player_features['continent']}
        Language: {player_features['language']}
        Religion: {player_features['religion']}
        Gender: {player_features['gender']}
        Country: {player_features['country']}
        Native Status: {player_features['native']}
        Care: {describe_trait_score(player_features['care'])}
        Fairness: {describe_trait_score(player_features['fairness'])}
        Loyalty: {describe_trait_score(player_features['loyalty'])}
        Authority: {describe_trait_score(player_features['authority'])}
        Sanctity: {describe_trait_score(player_features['sanctity'])}
        Conscientiousness: {describe_trait_score(player_features['conscientiousness'])}
        Extraversion: {describe_trait_score(player_features['extraversion'])}
        Neuroticism: {describe_trait_score(player_features['neuroticism'])}
        Openness: {describe_trait_score(player_features['openness'])}
        Agreeableness: {describe_trait_score(player_features['agreeableness'])}
    """
    return features_string.strip()

def compute_player_embedding(player_features: Dict) -> List[float]:
    """
    Compute the embedding of the player using the OpenAI API.
    """
    features_string = _player_features_to_string(player_features)
    response = client.embeddings.create(input=features_string, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding


def save_players(players: Optional[Dict[str, List[float]]] = None):
    """Save player vectors to file. If players is None, saves the global cache."""
    if players is None:
        players = _PLAYERS_CACHE
    if players is not None:
        with open(PLAYERS_PATH, "w", encoding="utf-8") as f:
            json.dump(players, f)


def get_player_vector(player_features: Dict) -> List[float]:
    """
    Get the player vector using the OpenAI API.
    If the vector is cached in memory, uses the cached value.
    Otherwise, computes, caches, and returns it.
    
    To compute the vector, we rely on extensive information provided about the player,
    including demographics, personality, political leaning, and more.
    """
    print(f"Getting player vector for features: {player_features}")
    player_key = _player_features_to_string(player_features)
    players = load_players()
    if player_key in players:
        return players[player_key]
 
    # Compute new player embedding
    player_embedding = compute_player_embedding(player_features)
    players[player_key] = player_embedding
    print(f"Computed player vector for features hash: {hash(player_key)}")
    save_players(players)
    return player_embedding

def get_player_vectors_batch(player_features_list: List[Dict], batch_size: int = 100) -> Dict[str, List[float]]:
    """
    Get player vectors for a list of player features in batches.
    Returns a dictionary mapping player feature strings to vectors.
    Only computes vectors for players not already cached.
    """
    players = load_players()
    results = {}

    # Separate cached and uncached players
    uncached_features = []
    uncached_keys = []
    for features in player_features_list:
        print(f"Getting player vector for features: {features}")
        player_key = _player_features_to_string(features)
        if player_key in players:
            results[player_key] = players[player_key]
        else:
            uncached_features.append(features)
            uncached_keys.append(player_key)

    # Process uncached players in batches
    for i in range(0, len(uncached_features), batch_size):
        batch_features = uncached_features[i:i + batch_size]
        batch_keys = uncached_keys[i:i + batch_size]
        batch_strings = [_player_features_to_string(f) for f in batch_features]
        print(f"Computing player vectors for batch {i//batch_size + 1}: {len(batch_features)} players")

        try:
            response = client.embeddings.create(input=batch_strings, model="text-embedding-3-small")
            for key, embedding_data in zip(batch_keys, response.data):
                players[key] = embedding_data.embedding
                results[key] = embedding_data.embedding
        except Exception as e:
            print(f"Error computing batch player vectors: {e}")
            # Fallback to individual requests for failed batch
            for features in batch_features:
                player_key = _player_features_to_string(features)
                if player_key not in results:
                    try:
                        results[player_key] = get_player_vector(features)
                    except Exception as e2:
                        print(f"Failed to get player vector for features hash {hash(player_key)}: {e2}")
                        raise

    # Save updated player vectors
    save_players(players)
    return results

def preload_player_vectors(player_features_list: List[Dict], batch_size: int = 100) -> None:
    """Preload player vectors for a list of player features into the cache."""
    get_player_vectors_batch(player_features_list, batch_size)

def clear_cache() -> None:
    """Clear the in-memory player vectors cache."""
    global _PLAYERS_CACHE
    _PLAYERS_CACHE = None