#!/usr/bin/env python3
"""
Interactive Data Explorer
Quickly explore specific features from the command line
"""

import pandas as pd
import sys

def load_data():
    """Load both preprocessed datasets."""
    gg_df = pd.read_csv('data/preprocessed/generate_guess.csv')
    cg_df = pd.read_csv('data/preprocessed/clue_generation.csv')
    return gg_df, cg_df

def explore_feature(df, feature, role='giver', dataset_name=''):
    """Explore a specific feature."""
    col_name = f"{role}.{feature}"
    
    if col_name not in df.columns:
        print(f"Error: Column '{col_name}' not found in dataset")
        print(f"Available columns: {[c for c in df.columns if c.startswith(role)]}")
        return
    
    print(f"\n{'='*70}")
    print(f"{dataset_name} - {role.upper()}.{feature.upper()}")
    print(f"{'='*70}")
    
    # Check if numeric or categorical
    if df[col_name].dtype in ['float64', 'int64']:
        # Numeric feature
        stats = df[col_name].describe()
        print(f"\nStatistics:")
        print(f"  Count:   {stats['count']:.0f}")
        print(f"  Mean:    {stats['mean']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        print(f"  Min:     {stats['min']:.2f}")
        print(f"  25%:     {stats['25%']:.2f}")
        print(f"  Median:  {stats['50%']:.2f}")
        print(f"  75%:     {stats['75%']:.2f}")
        print(f"  Max:     {stats['max']:.2f}")
        print(f"  Missing: {df[col_name].isna().sum()}")
    else:
        # Categorical feature
        value_counts = df[col_name].value_counts()
        total = len(df)
        
        print(f"\nDistribution:")
        print(f"  Total records: {total}")
        print(f"  Unique values: {len(value_counts)}")
        print(f"  Missing values: {df[col_name].isna().sum()}")
        print(f"\n  Top values:")
        for idx, (value, count) in enumerate(value_counts.head(10).items(), 1):
            pct = (count / total * 100)
            print(f"    {idx}. {value}: {count} ({pct:.2f}%)")

def list_features():
    """List available features."""
    print("\nAvailable Features:")
    print("\nCategorical:")
    print("  - marriage, education, race, continent, language")
    print("  - religion, gender, country, political")
    print("\nNumeric:")
    print("  - age")
    print("  Moral Foundations:")
    print("    - care, fairness, loyalty, authority, sanctity")
    print("  Big Five Personality:")
    print("    - conscientiousness, extraversion, neuroticism")
    print("    - openness, agreeableness")
    print("\nRoles: giver, guesser")
    print("Datasets: gg (Generate Guess), cg (Clue Generation)")

def main():
    """Main interactive explorer."""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print("Data Explorer - Interactive feature exploration tool")
        print("\nUsage:")
        print("  python3 data_analysis/explore.py <feature> [role] [dataset]")
        print("\nExamples:")
        print("  python3 data_analysis/explore.py age")
        print("  python3 data_analysis/explore.py education giver gg")
        print("  python3 data_analysis/explore.py care guesser cg")
        print("  python3 data_analysis/explore.py --list")
        list_features()
        return
    
    if sys.argv[1] == '--list':
        list_features()
        return
    
    # Parse arguments
    feature = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else 'giver'
    dataset = sys.argv[3] if len(sys.argv) > 3 else 'both'
    
    # Load data
    print("Loading datasets...")
    gg_df, cg_df = load_data()
    
    # Explore feature
    if dataset in ['gg', 'both']:
        explore_feature(gg_df, feature, role, 'Generate Guess')
    
    if dataset in ['cg', 'both']:
        explore_feature(cg_df, feature, role, 'Clue Generation')

if __name__ == "__main__":
    main()
