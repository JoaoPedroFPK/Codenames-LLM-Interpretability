import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_embedding import get_embedding
from utils.get_player_vector import get_player_vector
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import os
import numpy as np
import ast


def generate_guess_and_evaluate(batch_size=50, max_rows=1000, resume_from_batch=None):
    """
   This method will calculate the embeddings for each word and the hint. 
   It will then calculate cosine similarity between the embedding of the hint and the embeddings of the words.
   Then, the word with the highest target score will be considered as the AI Guess. 
   We will then evaluate if the AI guess corresponds to the player's guess.
   
   Args:
       batch_size: Number of rows to process before saving
       max_rows: Maximum number of rows to process
       resume_from_batch: Batch number to resume from (None to start fresh)
    """

    df = pd.read_csv('data/preprocessed/generate_guess.csv')
    
    # Setup output directory and base filename
    os.makedirs('data/experiments', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_path = f'data/experiments/generate_guess_evaluation_{timestamp}'
    final_output_path = f'{base_output_path}.csv'
    
    all_results = []
    batch_results = []
    
    # Calculate starting point
    start_row = 0
    if resume_from_batch:
        start_row = (resume_from_batch - 1) * batch_size
        print(f"Resuming from batch {resume_from_batch}, starting at row {start_row}")
    
    # Process data in batches
    total_rows = min(len(df), max_rows)
    processed_rows = start_row
    
    print(f"Processing rows {start_row} to {total_rows} in batches of {batch_size}")
    
    for index, row in df.iloc[start_row:max_rows].iterrows():
        remaining = ast.literal_eval(row['remaining'])
        print(f"Remaining: {remaining}")
        hint = row['hint']
        print(f"Hint: {hint}")
        player_guess = row['output']
        print(f"Player guess: {player_guess}")

        hint_embedding = get_embedding(hint)
        # Remaining words are normalized by the board. The user does not see each word individually, but rather the board as a whole.
        remaining_embeddings = [get_embedding(word) for word in remaining]

        # Ensure all embeddings have the same dimension by truncating to minimum length
        all_embeddings = [hint_embedding] + remaining_embeddings
        min_dim = min(len(emb) for emb in all_embeddings)
        
        hint_embedding = hint_embedding[:min_dim]
        remaining_embeddings = [emb[:min_dim] for emb in remaining_embeddings]

        target_scores = []
        for word, embedding in zip(remaining, remaining_embeddings):
            # Reshape to 2D arrays for cosine_similarity
            hint_2d = np.array(hint_embedding).reshape(1, -1)
            embedding_2d = np.array(embedding).reshape(1, -1)
            target_score = cosine_similarity(hint_2d, embedding_2d)[0][0]
            target_scores.append(target_score)

        # Normalize target scores relative to other scores for this hint
        target_scores_array = np.array(target_scores)
        mean_target_score = np.mean(target_scores_array)
        std_target_score = np.std(target_scores_array)
        normalized_target_scores = (target_scores_array - mean_target_score) / (std_target_score + 1e-6)

        max_score = np.max(normalized_target_scores)
        ai_guess = remaining[np.argmax(normalized_target_scores)]
        
        is_ai_aligned_with_human_guess = ai_guess == player_guess
        
        result_obj = {
            'hint': hint,
            'remaining': remaining,
            'player_guess': player_guess,
            'ai_guess': ai_guess,
            'max_normalized_cosine_similarity': max_score,
            'is_aligned': is_ai_aligned_with_human_guess,
            'target_scores': target_scores,
            'normalized_target_scores': normalized_target_scores.tolist()
        }
        batch_results.append(result_obj)
        all_results.append(result_obj)
        processed_rows += 1
        
        # Save batch when batch_size is reached
        if len(batch_results) >= batch_size:
            batch_num = processed_rows // batch_size
            batch_output_path = f'{base_output_path}_batch_{batch_num}.csv'
            batch_df = pd.DataFrame(batch_results)
            batch_df.to_csv(batch_output_path, index=False)
            print(f"Batch {batch_num} saved: {len(batch_results)} rows -> {batch_output_path}")
            batch_results = []  # Reset batch
        
        # Progress update
        if processed_rows % 10 == 0:
            print(f"Progress: {processed_rows}/{total_rows} rows processed")
    
    # Save any remaining results in the final batch
    if batch_results:
        batch_num = (processed_rows - 1) // batch_size + 1
        batch_output_path = f'{base_output_path}_batch_{batch_num}.csv'
        batch_df = pd.DataFrame(batch_results)
        batch_df.to_csv(batch_output_path, index=False)
        print(f"Final batch {batch_num} saved: {len(batch_results)} rows -> {batch_output_path}")
    
    # Create final consolidated file
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(final_output_path, index=False)
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total rows processed: {len(all_results)}")
        print(f"Final results saved to: {final_output_path}")
        print(f"Alignment rate: {results_df['is_aligned'].mean():.2%}")
        
        return results_df
    else:
        print("No results to save.")
        return pd.DataFrame()


if __name__ == "__main__":
    # You can adjust batch_size and max_rows as needed
    generate_guess_and_evaluate(batch_size=50, max_rows=1000)