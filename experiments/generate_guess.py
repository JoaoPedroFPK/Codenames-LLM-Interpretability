import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_embedding import get_embedding, get_embeddings_batch, preload_embeddings
from utils.get_player_vector import get_player_vector, preload_player_vectors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import numpy as np
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import multiprocessing


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


def collect_unique_texts(df: pd.DataFrame, max_rows: Optional[int] = None) -> set:
    """
    Collect all unique words and hints from the dataset.

    Args:
        df: DataFrame containing the dataset
        max_rows: Maximum number of rows to process (None for all rows)

    Returns:
        set: Set of unique texts (words and hints)
    """
    unique_texts = set()

    rows_to_process = df.head(max_rows) if max_rows else df

    for _, row in rows_to_process.iterrows():
        # Add hint
        unique_texts.add(str(row['hint']).strip())

        # Add all remaining words
        remaining = ast.literal_eval(row['remaining'])
        for word in remaining:
            unique_texts.add(str(word).strip())

    return unique_texts


def preload_all_embeddings(df: pd.DataFrame, max_rows: Optional[int] = None, batch_size: int = 100) -> None:
    """
    Preload embeddings for all unique words and hints in the dataset.

    Args:
        df: DataFrame containing the dataset
        max_rows: Maximum number of rows to process (None for all rows)
        batch_size: Batch size for embedding computation
    """
    print("Collecting unique texts from dataset...")
    unique_texts = collect_unique_texts(df, max_rows)
    print(f"Found {len(unique_texts)} unique texts to embed")

    print("Preloading embeddings in batches...")
    preload_embeddings(list(unique_texts), batch_size=batch_size)
    print("Embeddings preloaded successfully")


def calculate_embeddings(hint, remaining_words):
    """
    Calculate embeddings for the hint and all remaining words on the board.
    Assumes embeddings have been preloaded into cache.

    Args:
        hint (str): The hint given to players
        remaining_words (list): List of remaining words on the board

    Returns:
        tuple: (hint_embedding, remaining_embeddings) where embeddings are normalized to same dimension
    """
    hint_embedding = get_embedding(hint)  # Will use cached embedding
    remaining_embeddings = [get_embedding(word) for word in remaining_words]  # All cached

    # Ensure all embeddings have the same dimension by truncating to minimum length
    all_embeddings = [hint_embedding] + remaining_embeddings
    min_dim = min(len(emb) for emb in all_embeddings)

    hint_embedding = hint_embedding[:min_dim]
    remaining_embeddings = [emb[:min_dim] for emb in remaining_embeddings]

    return hint_embedding, remaining_embeddings


def compute_cosine_similarities(hint_embedding, word_embeddings, player_embedding=None, alpha=0.3):
    """
    Compute cosine similarity between hint embedding and each word embedding.
    If player_embedding is provided, uses score fusion: 
    score = (1 - alpha) * cos_sim(hint, word) + alpha * cos_sim(player, word)
    
    Args:
        hint_embedding (list): Embedding vector for the hint
        word_embeddings (list): List of embedding vectors for words
        player_embedding (list, optional): Embedding vector for the player
        alpha (float): Weight for player embedding in score fusion (default: 0.3)
        
    Returns:
        list: Cosine similarity scores for each word
    """
    target_scores = []
    
    for embedding in word_embeddings:
        # Reshape to 2D arrays for cosine_similarity
        hint_2d = np.array(hint_embedding).reshape(1, -1)
        embedding_2d = np.array(embedding).reshape(1, -1)
        hint_score = cosine_similarity(hint_2d, embedding_2d)[0][0]
        
        if player_embedding is not None:
            # Apply score fusion with player embedding
            player_2d = np.array(player_embedding).reshape(1, -1)
            player_score = cosine_similarity(player_2d, embedding_2d)[0][0]
            target_score = (1 - alpha) * hint_score + alpha * player_score
        else:
            target_score = hint_score
            
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


def process_single_row(row_data: Dict[str, Any], use_player_context: bool = False, alpha: float = 0.3) -> Dict[str, Any]:
    """
    Process a single row of data to compute AI guess and alignment.

    Args:
        row_data: Dictionary containing row data
        use_player_context: Whether to include player embedding in score calculation
        alpha: Weight for player embedding in score fusion (only used if use_player_context=True)

    Returns:
        dict: Result object with all evaluation metrics
    """
    remaining = row_data['remaining']
    hint = row_data['hint']
    player_guess = row_data['player_guess']
    giver_features = row_data.get('giver_features')

    # Calculate embeddings for hint and remaining words (should be cached)
    hint_embedding, remaining_embeddings = calculate_embeddings(hint, remaining)

    # Get player embedding if use_player_context is True
    player_embedding = None
    if use_player_context and giver_features:
        player_embedding = get_player_vector(giver_features)
        # Normalize player embedding to same dimension as hint/word embeddings
        min_dim = len(hint_embedding)
        player_embedding = player_embedding[:min_dim]

    # Compute cosine similarities between hint and each word (with optional player context)
    target_scores = compute_cosine_similarities(hint_embedding, remaining_embeddings, player_embedding, alpha)

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

    return result_obj


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
    If player_guess contains a comma, split and pick only the first word (top-1 approach).

    Args:
        ai_guess (str): The word predicted by the AI
        player_guess (str): The word chosen by the human player

    Returns:
        bool: True if AI and human guesses match, False otherwise
    """
    if isinstance(player_guess, str) and "," in player_guess:
        # Split by comma, remove whitespace, and pick the first guess
        guesses = [g.strip() for g in player_guess.split(",")]
        top_player_guess = guesses[0] if guesses else player_guess
    else:
        top_player_guess = player_guess

    return ai_guess == top_player_guess


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

def extract_player_features(df, prefix: str) -> dict:
    """
    Extracts player-specific features from a pandas DataFrame using a provided column prefix.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing player data with column names prefixed (e.g., "giver.age").
        prefix (str): The prefix in the column names corresponding to the player ("giver" or "guesser").
    
    Returns:
        list of dict: A list of dictionaries where each dictionary represents the selected features for a player in a row.
                      The dictionary keys are feature names (e.g., "age", "care"), and values are the corresponding values from the DataFrame.
    """
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
    cols = {f"{prefix}.{feat}": feat for feat in PLAYER_FEATURES}
    return df[list(cols.keys())].rename(columns=cols).to_dict(orient="records")




def generate_guess_experiment(batch_size=50, max_rows=1000, resume_from_batch=None, max_workers=None, use_player_context=False, alpha=0.3):
    """
    Main function to generate AI guesses and evaluate alignment with human players.

    This method preloads embeddings for all unique words/hints, then uses parallel processing
    to calculate embeddings similarities and evaluate alignment with human players.

    Args:
        batch_size (int): Number of rows to process before saving intermediate results
        max_rows (int): Maximum number of rows to process from the dataset
        resume_from_batch (int, optional): Batch number to resume from (None to start fresh)
        max_workers (int, optional): Maximum number of worker processes (defaults to CPU count)
        use_player_context (bool): Whether to use player embedding in score calculation (default: False)
        alpha (float): Weight for player embedding in score fusion (default: 0.3, only used if use_player_context=True)

    Returns:
        pandas.DataFrame: DataFrame containing all evaluation results with alignment metrics
    """
    # Load data and setup output paths
    df, base_output_path, final_output_path = load_and_setup_data()

    # Preload all embeddings first
    preload_all_embeddings(df, max_rows, batch_size=100)

    # Preload player embeddings if using player context
    if use_player_context:
        print("Extracting and preloading giver player embeddings...")
        giver_features_list = extract_player_features(df.head(max_rows) if max_rows else df, prefix="giver")
        preload_player_vectors(giver_features_list, batch_size=100)
        print("Player embeddings preloaded successfully")

    # Set up parallel processing
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 to respect API limits

    all_results = []
    batch_results = []

    # Calculate starting point for resuming
    start_row = 0
    if resume_from_batch:
        start_row = (resume_from_batch - 1) * batch_size
        print(f"Resuming from batch {resume_from_batch}, starting at row {start_row}")

    # Prepare data for parallel processing
    total_rows = min(len(df), max_rows)
    rows_to_process = df.iloc[start_row:max_rows]

    print(f"Processing rows {start_row} to {total_rows} using {max_workers} workers")
    if use_player_context:
        print(f"Using player context with alpha={alpha}")

    # Process rows in parallel batches
    processed_rows = start_row

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for index, row in rows_to_process.iterrows():
            # Extract giver features if using player context
            giver_features = None
            if use_player_context:
                giver_features_list = extract_player_features(pd.DataFrame([row]), prefix="giver")
                giver_features = giver_features_list[0] if giver_features_list else None
            
            row_data = {
                'remaining': ast.literal_eval(row['remaining']),
                'hint': row['hint'],
                'player_guess': row['output'],
                'giver_features': giver_features
            }
            future = executor.submit(process_single_row, row_data, use_player_context, alpha)
            future_to_index[future] = index

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_obj = future.result()
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

            except Exception as exc:
                print(f'Row {index} generated an exception: {exc}')
                raise

    # Save any remaining results in the final batch
    if batch_results:
        batch_num = (processed_rows - 1) // batch_size + 1
        save_batch_results(batch_results, batch_num, base_output_path)

    # Create final consolidated file and return results
    return save_final_results(all_results, final_output_path)





if __name__ == "__main__":
    # You can adjust batch_size and max_rows as needed
    # Set use_player_context=True to include player personality in scoring
    generate_guess_experiment(batch_size=50, max_rows=1000, use_player_context=False, alpha=0.3)