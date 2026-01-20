


def generate_guess_and_evaluate(data):
    """
   This method will calculate the embeddings for each word and the hint. 
   It will then calculate cosine similarity between the embedding of the hint and the embeddings of the words.
   Then, the word with the highest target score will be considered as the AI Guess. 
   We will then evaluate if the AI guess corresponds to the player's guess. 
    """

    for index, row in data.iterrows():
        remaining = row['remaining']
        hint = row['hint']
        player_guess = row['output']






def generate_guess_and_evaluate_with_context():
    """
    This method will calculate the embeddings for each word and the hint. 
    It will then calculate cosine similarity between the embedding of the hint and the embeddings of the words.
    Then, the word with the highest target score will be considered as the AI Guess. 
    We will then evaluate if the AI guess corresponds to the player's guess. 
    However, this time the AI will also use as context the demographics, the political leaning, the event, the demographics only, the personality only, and all the information from the Giver.
    We will measure whether providing the context approximates the embedding of the player's guess.
    """
    pass





