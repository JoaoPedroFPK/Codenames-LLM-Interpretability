from tarfile import data_filter
from utils.get_embedding import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import os


def generate_guess_and_evaluate():
    """
   This method will calculate the embeddings for each word and the hint. 
   It will then calculate cosine similarity between the embedding of the hint and the embeddings of the words.
   Then, the word with the highest target score will be considered as the AI Guess. 
   We will then evaluate if the AI guess corresponds to the player's guess. 
    """

    df = pd.read_csv('data/preprocessed/generate_guess.csv')
    
    results = []

    for index, row in df.iterrows():
        remaining = row['remaining']
        hint = row['hint']
        player_guess = row['output']

        hint_embedding = get_embedding(hint)
        remaining_embeddings = [get_embedding(word) for word in remaining]

        target_scores = []
        for word, embedding in zip(remaining, remaining_embeddings):
            target_score = cosine_similarity(hint_embedding, embedding)
            target_scores.append(target_score)

        max_score = max(target_scores)
        ai_guess = remaining[target_scores.index(max_score)]
        
        is_ai_aligned_with_human_guess = ai_guess == player_guess
        
        result_obj = {
            'hint': hint,
            'remaining': remaining,
            'player_guess': player_guess,
            'ai_guess': ai_guess,
            'max_cosine_similarity': max_score,
            'is_aligned': is_ai_aligned_with_human_guess,
            'target_scores': target_scores
        }
        results.append(result_obj)
    
    results_df = pd.DataFrame(results)
    
    os.makedirs('data/experiments', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'data/experiments/generate_guess_evaluation_{timestamp}.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"Evaluation results saved to {output_path}")
    print(f"Alignment rate: {results_df['is_aligned'].mean():.2%}")
    
    return results_df




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




