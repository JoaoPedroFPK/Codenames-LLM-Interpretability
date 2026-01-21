from tarfile import data_filter
from utils.get_embedding import get_embedding
from utils.get_player_vector import get_player_vector
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import os




PLAYER_FEATURES = [
    "age",
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity",
    "political",
    "conscientiousness",
    "extraversion",
    "neuroticism",
    "openness",
    "agreeableness",
]


def extract_player_features(df, prefix: str) -> dict:
    cols = {f"{prefix}.{feat}": feat for feat in PLAYER_FEATURES}
    return df[list(cols.keys())].rename(columns=cols).to_dict(orient="records")



def generate_guess_and_evaluate_with_context():
    """
    This method will calculate the embeddings for each word and the hint. 
    It will then calculate cosine similarity between the embedding of the hint and the embeddings of the words.
    Then, the word with the highest target score will be considered as the AI Guess. 
    We will then evaluate if the AI guess corresponds to the player's guess. 
    However, this time the AI will also use as context the demographics, the political leaning, the event, the demographics only, the personality only, and all the information from the Giver.
    We will measure whether providing the context approximates the embedding of the player's guess.
    """
    

    df = pd.read_csv('data/preprocessed/generate_guess.csv')
    results = []

    for index, row in df.iterrows():
        remaining = row['remaining']
        hint = row['hint']
        player_guess = row['output']
        giver_features = extract_player_features(df, "giver")
        guesser_features = extract_player_features(df, "guesser")

        giver_embedding = get_player_vector(giver_features)
        guesser_embedding = get_player_vector(guesser_features)
        