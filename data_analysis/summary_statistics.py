import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load both preprocessed datasets."""
    print("Loading datasets...")
    gg_df = pd.read_csv('data/preprocessed/generate_guess.csv')
    cg_df = pd.read_csv('data/preprocessed/clue_generation.csv')
    return gg_df, cg_df

def generate_summary_stats(df, dataset_name):
    """Generate comprehensive summary statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} - SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Categorical features
    categorical_features = ['marriage', 'education', 'race', 'continent', 'language', 
                          'religion', 'gender', 'country', 'political']
    
    # Numeric features
    numeric_features = ['age', 'care', 'fairness', 'loyalty', 'authority', 'sanctity',
                       'conscientiousness', 'extraversion', 'neuroticism', 'openness', 
                       'agreeableness']
    
    for role in ['giver', 'guesser']:
        print(f"\n{'-'*80}")
        print(f"{role.upper()} STATISTICS")
        print(f"{'-'*80}")
        
        # Categorical features summary
        print(f"\n{role.upper()} - Categorical Features:")
        print(f"{'-'*60}")
        
        for feature in categorical_features:
            col_name = f"{role}.{feature}"
            if col_name in df.columns:
                unique_count = df[col_name].nunique()
                most_common = df[col_name].mode()[0] if len(df[col_name].mode()) > 0 else "N/A"
                most_common_count = df[col_name].value_counts().iloc[0] if len(df[col_name]) > 0 else 0
                most_common_pct = (most_common_count / len(df) * 100) if len(df) > 0 else 0
                missing = df[col_name].isna().sum()
                
                print(f"\n{feature.upper()}:")
                print(f"  Unique values: {unique_count}")
                print(f"  Most common: {most_common} ({most_common_count} occurrences, {most_common_pct:.2f}%)")
                print(f"  Missing values: {missing}")
                
                # Show top 5 categories
                print(f"  Top 5 categories:")
                top_5 = df[col_name].value_counts().head(5)
                for category, count in top_5.items():
                    pct = (count / len(df) * 100)
                    print(f"    {category}: {count} ({pct:.2f}%)")
        
        # Numeric features summary
        print(f"\n{role.upper()} - Numeric Features:")
        print(f"{'-'*60}")
        
        for feature in numeric_features:
            col_name = f"{role}.{feature}"
            if col_name in df.columns:
                stats = df[col_name].describe()
                missing = df[col_name].isna().sum()
                
                print(f"\n{feature.upper()}:")
                print(f"  Count: {stats['count']:.0f}")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  25%: {stats['25%']:.2f}")
                print(f"  Median: {stats['50%']:.2f}")
                print(f"  75%: {stats['75%']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
                print(f"  Missing: {missing}")

def generate_comparison_report(df, dataset_name):
    """Generate comparison report between giver and guesser."""
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} - GIVER vs GUESSER COMPARISON")
    print(f"{'='*80}")
    
    # Numeric features comparison
    numeric_features = ['age', 'care', 'fairness', 'loyalty', 'authority', 'sanctity',
                       'conscientiousness', 'extraversion', 'neuroticism', 'openness', 
                       'agreeableness']
    
    print(f"\nNumeric Features Comparison:")
    print(f"{'-'*80}")
    print(f"{'Feature':<20} {'Giver Mean':<15} {'Guesser Mean':<15} {'Difference':<15}")
    print(f"{'-'*80}")
    
    for feature in numeric_features:
        giver_col = f"giver.{feature}"
        guesser_col = f"guesser.{feature}"
        
        if giver_col in df.columns and guesser_col in df.columns:
            giver_mean = df[giver_col].mean()
            guesser_mean = df[guesser_col].mean()
            diff = giver_mean - guesser_mean
            
            print(f"{feature:<20} {giver_mean:<15.2f} {guesser_mean:<15.2f} {diff:+.2f}")
    
    # Categorical features comparison
    categorical_features = ['marriage', 'education', 'race', 'continent', 'language', 
                          'religion', 'gender', 'country', 'political']
    
    print(f"\n\nCategorical Features - Most Common Values:")
    print(f"{'-'*80}")
    print(f"{'Feature':<20} {'Giver Most Common':<25} {'Guesser Most Common':<25}")
    print(f"{'-'*80}")
    
    for feature in categorical_features:
        giver_col = f"giver.{feature}"
        guesser_col = f"guesser.{feature}"
        
        if giver_col in df.columns and guesser_col in df.columns:
            giver_mode = df[giver_col].mode()[0] if len(df[giver_col].mode()) > 0 else "N/A"
            guesser_mode = df[guesser_col].mode()[0] if len(df[guesser_col].mode()) > 0 else "N/A"
            
            print(f"{feature:<20} {str(giver_mode):<25} {str(guesser_mode):<25}")

def save_summary_to_file(gg_df, cg_df):
    """Save summary statistics to a text file."""
    output_dir = Path('output/data_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / 'summary_statistics.txt'
    
    import sys
    from io import StringIO
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Generate all summaries
    generate_summary_stats(gg_df, "Generate Guess Dataset")
    generate_comparison_report(gg_df, "Generate Guess Dataset")
    
    generate_summary_stats(cg_df, "Clue Generation Dataset")
    generate_comparison_report(cg_df, "Clue Generation Dataset")
    
    # Get the output
    output = mystdout.getvalue()
    sys.stdout = old_stdout
    
    # Save to file
    with open(summary_file, 'w') as f:
        f.write(output)
    
    print(output)
    print(f"\nSummary statistics saved to: {summary_file}")

def main():
    """Main function."""
    print("="*80)
    print("CODENAMES LLM INTERPRETABILITY - SUMMARY STATISTICS")
    print("="*80)
    
    # Load data
    gg_df, cg_df = load_data()
    
    # Generate and save summaries
    save_summary_to_file(gg_df, cg_df)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS GENERATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
