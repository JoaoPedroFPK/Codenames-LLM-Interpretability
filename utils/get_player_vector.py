from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

PLAYERS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "players.json"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_players():
    if os.path.exists(PLAYERS_PATH):
        with open(PLAYERS_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                return {}
    else:
        return {}


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



def compute_player_embedding(player_features):
    """
    This method will compute the embedding of the player using the OpenAI API.
    """

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

    response = client.embeddings.create(input=features_string, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding


def save_players(players):
    with open(PLAYERS_PATH, "w", encoding="utf-8") as f:
        json.dump(players, f)


def get_player_vector(player_features):
    """
    This method will either load the player already computed vector or compute it and save it.
    To compute the vector, we will rely on the extensive information provided about him. 
    This information includes demographics, personality, political leaning, event, and more.
    We will use OpenAI's Embedding model to compute the vector. Then, this vector will be concatenated 
    with the embedding of the hint to form the player's vector.
    """

    players = load_players()
    if player_features in players:
        return players[player_features]

 
    player_embedding = compute_player_embedding(player_features) 

    players[player_features] = player_embedding
    save_players(players)
    return player_embedding