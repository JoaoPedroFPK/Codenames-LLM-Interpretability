import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_embedding import get_embedding
from utils.get_player_vector import get_player_vector
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import numpy as np
import ast


def load_and_setup_data():
    """
    Load the preprocessed generate_guess data and setup output directory structure.
    
    Returns:
        tuple: (DataFrame, str, str) containing the loaded data, base output path, and final output path
    """
    df = pd.read_csv('data/preprocessed/generate_guess.csv')
    
    # Setup output directory and base filename
    os.makedirs('data/experiments', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_path = f'data/experiments/generate_guess_evaluation_{timestamp}'
    final_output_path = f'{base_output_path}.csv'
    
    return df, base_output_path, final_output_path


def calculate_embeddings(hint, remaining_words):
    """
    Calculate embeddings for the hint and all remaining words on the board.
    
    Args:
        hint (str): The hint given to players
        remaining_words (list): List of remaining words on the board
        
    Returns:
        tuple: (hint_embedding, remaining_embeddings) where embeddings are normalized to same dimension
    """
    hint_embedding = get_embedding(hint)
    remaining_embeddings = [get_embedding(word) for word in remaining_words]
    
    # Ensure all embeddings have the same dimension by truncating to minimum length
    all_embeddings = [hint_embedding] + remaining_embeddings
    min_dim = min(len(emb) for emb in all_embeddings)
    
    hint_embedding = hint_embedding[:min_dim]
    remaining_embeddings = [emb[:min_dim] for emb in remaining_embeddings]
    
    return hint_embedding, remaining_embeddings


def compute_cosine_similarities(hint_embedding, word_embeddings):
    """
    Compute cosine similarity between hint embedding and each word embedding.
    
    Args:
        hint_embedding (list): Embedding vector for the hint
        word_embeddings (list): List of embedding vectors for words
        
    Returns:
        list: Cosine similarity scores for each word
    """
    target_scores = []
    
    for embedding in word_embeddings:
        # Reshape to 2D arrays for cosine_similarity
        hint_2d = np.array(hint_embedding).reshape(1, -1)
        embedding_2d = np.array(embedding).reshape(1, -1)
        target_score = cosine_similarity(hint_2d, embedding_2d)[0][0]
        target_scores.append(target_score)
    
    return target_scores


def normalize_scores(scores):
    """
    Normalize target scores using z-score normalization relative to other scores for the same hint.
    
    Args:
        scores (list): Raw cosine similarity scores
        
    Returns:
        numpy.ndarray: Normalized scores
    """
    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    normalized_scores = (scores_array - mean_score) / (std_score + 1e-6)
    
    return normalized_scores


def predict_ai_guess(remaining_words, normalized_scores):
    """
    Predict the AI's guess based on the highest normalized similarity score.
    
    Args:
        remaining_words (list): List of remaining words on the board
        normalized_scores (numpy.ndarray): Normalized similarity scores
        
    Returns:
        tuple: (ai_guess, max_score) where ai_guess is the predicted word and max_score is the highest score
    """
    max_score = np.max(normalized_scores)
    ai_guess = remaining_words[np.argmax(normalized_scores)]
    
    return ai_guess, max_score


def evaluate_alignment(ai_guess, player_guess):
    """
    Evaluate whether the AI's guess aligns with the human player's guess.
    
    Args:
        ai_guess (str): The word predicted by the AI
        player_guess (str): The word chosen by the human player
        
    Returns:
        bool: True if AI and human guesses match, False otherwise
    """
    return ai_guess == player_guess


def create_result_object(hint, remaining, player_guess, ai_guess, target_scores, normalized_scores, alignment):
    """
    Create a comprehensive result object containing all evaluation metrics.
    
    Args:
        hint (str): The hint given to players
        remaining (list): List of remaining words on the board
        player_guess (str): The human player's guess
        ai_guess (str): The AI's predicted guess
        target_scores (list): Raw cosine similarity scores
        normalized_scores (numpy.ndarray): Normalized similarity scores
        alignment (bool): Whether AI and human guesses align
        
    Returns:
        dict: Comprehensive result object with all metrics
    """
    max_score = np.max(normalized_scores)
    
    return {
        'hint': hint,
        'remaining': remaining,
        'player_guess': player_guess,
        'ai_guess': ai_guess,
        'max_normalized_cosine_similarity': max_score,
        'is_aligned': alignment,
        'target_scores': target_scores,
        'normalized_target_scores': normalized_scores.tolist()
    }


def save_batch_results(results, batch_num, base_output_path):
    """
    Save a batch of results to a CSV file.
    
    Args:
        results (list): List of result dictionaries
        batch_num (int): Batch number for filename
        base_output_path (str): Base path for output files
        
    Returns:
        str: Path to the saved batch file
    """
    batch_output_path = f'{base_output_path}_batch_{batch_num}.csv'
    batch_df = pd.DataFrame(results)
    batch_df.to_csv(batch_output_path, index=False)
    print(f"Batch {batch_num} saved: {len(results)} rows -> {batch_output_path}")
    
    return batch_output_path


def save_final_results(all_results, output_path):
    """
    Save all consolidated results to a final CSV file and print summary statistics.
    
    Args:
        all_results (list): List of all result dictionaries
        output_path (str): Path for the final output file
        
    Returns:
        pandas.DataFrame: DataFrame containing all results
    """
    if not all_results:
        print("No results to save.")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total rows processed: {len(all_results)}")
    print(f"Final results saved to: {output_path}")
    print(f"Alignment rate: {results_df['is_aligned'].mean():.2%}")
    
    return results_df


def generate_guess_and_evaluate(batch_size=50, max_rows=1000, resume_from_batch=None):
    """
    Main function to generate AI guesses and evaluate alignment with human players.
    
    This method calculates embeddings for each word and the hint, computes cosine similarity 
    between the hint embedding and word embeddings, then selects the word with the highest 
    normalized similarity score as the AI guess. Finally, it evaluates if the AI guess 
    corresponds to the player's guess.
    
    Args:
        batch_size (int): Number of rows to process before saving intermediate results
        max_rows (int): Maximum number of rows to process from the dataset
        resume_from_batch (int, optional): Batch number to resume from (None to start fresh)
        
    Returns:
        pandas.DataFrame: DataFrame containing all evaluation results with alignment metrics
    """
    # Load data and setup output paths
    df, base_output_path, final_output_path = load_and_setup_data()
    
    all_results = []
    batch_results = []
    
    # Calculate starting point for resuming
    start_row = 0
    if resume_from_batch:
        start_row = (resume_from_batch - 1) * batch_size
        print(f"Resuming from batch {resume_from_batch}, starting at row {start_row}")
    
    # Process data in batches
    total_rows = min(len(df), max_rows)
    processed_rows = start_row
    
    print(f"Processing rows {start_row} to {total_rows} in batches of {batch_size}")
    
    for index, row in df.iloc[start_row:max_rows].iterrows():
        # Extract data from current row
        remaining = ast.literal_eval(row['remaining'])
        hint = row['hint']
        player_guess = row['output']
        
        print(f"Remaining: {remaining}")
        print(f"Hint: {hint}")
        print(f"Player guess: {player_guess}")
        
        # Calculate embeddings for hint and remaining words
        hint_embedding, remaining_embeddings = calculate_embeddings(hint, remaining)
        
        # Compute cosine similarities between hint and each word
        target_scores = compute_cosine_similarities(hint_embedding, remaining_embeddings)
        
        # Normalize scores relative to other scores for this hint
        normalized_scores = normalize_scores(target_scores)
        
        # Predict AI's best guess
        ai_guess, max_score = predict_ai_guess(remaining, normalized_scores)
        
        # Evaluate alignment between AI and human guess
        alignment = evaluate_alignment(ai_guess, player_guess)
        
        # Create comprehensive result object
        result_obj = create_result_object(
            hint, remaining, player_guess, ai_guess, 
            target_scores, normalized_scores, alignment
        )
        
        batch_results.append(result_obj)
        all_results.append(result_obj)
        processed_rows += 1
        
        # Save batch when batch_size is reached
        if len(batch_results) >= batch_size:
            batch_num = processed_rows // batch_size
            save_batch_results(batch_results, batch_num, base_output_path)
            batch_results = []  # Reset batch
        
        # Progress update
        if processed_rows % 10 == 0:
            print(f"Progress: {processed_rows}/{total_rows} rows processed")
    
    # Save any remaining results in the final batch
    if batch_results:
        batch_num = (processed_rows - 1) // batch_size + 1
        save_batch_results(batch_results, batch_num, base_output_path)
    
    # Create final consolidated file and return results
    return save_final_results(all_results, final_output_path)


if __name__ == "__main__":
    # You can adjust batch_size and max_rows as needed
    generate_guess_and_evaluate(batch_size=50, max_rows=1000)