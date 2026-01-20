import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory for visualizations
output_dir = Path('output/data_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load both preprocessed datasets."""
    print("Loading datasets...")
    gg_df = pd.read_csv('data/preprocessed/generate_guess.csv')
    cg_df = pd.read_csv('data/preprocessed/clue_generation.csv')
    return gg_df, cg_df

def analyze_categorical_feature(df, feature_name, role='giver', dataset_name='', save_path=None):
    """Analyze and visualize distribution of a categorical feature."""
    col_name = f"{role}.{feature_name}"
    
    if col_name not in df.columns:
        print(f"Column {col_name} not found in dataset")
        return None
    
    # Calculate value counts and percentages
    value_counts = df[col_name].value_counts()
    percentages = (value_counts / len(df) * 100).round(2)
    
    stats = pd.DataFrame({
        'count': value_counts,
        'percentage': percentages
    })
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} - {role.upper()} {feature_name.upper()} Distribution")
    print(f"{'='*60}")
    print(stats)
    print(f"Total unique values: {len(value_counts)}")
    print(f"Total records: {len(df)}")
    print(f"Missing values: {df[col_name].isna().sum()}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    value_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title(f'{role.capitalize()} {feature_name.capitalize()} - Count Distribution\n({dataset_name})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(feature_name.capitalize(), fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    colors = sns.color_palette('pastel', len(value_counts))
    ax2.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax2.set_title(f'{role.capitalize()} {feature_name.capitalize()} - Percentage Distribution\n({dataset_name})', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()
    
    return stats

def analyze_numeric_feature(df, feature_name, role='giver', dataset_name='', save_path=None):
    """Analyze and visualize distribution of a numeric feature."""
    col_name = f"{role}.{feature_name}"
    
    if col_name not in df.columns:
        print(f"Column {col_name} not found in dataset")
        return None
    
    # Calculate statistics
    stats = df[col_name].describe()
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} - {role.upper()} {feature_name.upper()} Statistics")
    print(f"{'='*60}")
    print(stats)
    print(f"Missing values: {df[col_name].isna().sum()}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    df[col_name].dropna().hist(bins=30, ax=ax1, color='lightcoral', edgecolor='black')
    ax1.set_title(f'{role.capitalize()} {feature_name.capitalize()} - Distribution\n({dataset_name})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(feature_name.capitalize(), fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.axvline(df[col_name].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col_name].mean():.2f}')
    ax1.axvline(df[col_name].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col_name].median():.2f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    df[col_name].dropna().plot(kind='box', ax=ax2, vert=True, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
    ax2.set_title(f'{role.capitalize()} {feature_name.capitalize()} - Box Plot\n({dataset_name})', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel(feature_name.capitalize(), fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()
    
    return stats

def compare_roles(df, feature_name, dataset_name='', save_path=None):
    """Compare distribution of a feature between giver and guesser."""
    giver_col = f"giver.{feature_name}"
    guesser_col = f"guesser.{feature_name}"
    
    if giver_col not in df.columns or guesser_col not in df.columns:
        print(f"Columns not found for comparison")
        return None
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Comparing GIVER vs GUESSER: {feature_name.upper()}")
    print(f"{'='*60}")
    
    # Check if numeric or categorical
    if df[giver_col].dtype in ['float64', 'int64']:
        # Numeric comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histograms
        df[giver_col].dropna().hist(bins=30, ax=axes[0, 0], color='skyblue', 
                                    alpha=0.7, edgecolor='black', label='Giver')
        axes[0, 0].set_title(f'Giver {feature_name.capitalize()} Distribution', fontweight='bold')
        axes[0, 0].set_xlabel(feature_name.capitalize())
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        df[guesser_col].dropna().hist(bins=30, ax=axes[0, 1], color='lightcoral', 
                                      alpha=0.7, edgecolor='black', label='Guesser')
        axes[0, 1].set_title(f'Guesser {feature_name.capitalize()} Distribution', fontweight='bold')
        axes[0, 1].set_xlabel(feature_name.capitalize())
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Overlapped histogram
        axes[1, 0].hist([df[giver_col].dropna(), df[guesser_col].dropna()], 
                       bins=30, label=['Giver', 'Guesser'], 
                       color=['skyblue', 'lightcoral'], alpha=0.6, edgecolor='black')
        axes[1, 0].set_title(f'{feature_name.capitalize()} - Overlapped Comparison', fontweight='bold')
        axes[1, 0].set_xlabel(feature_name.capitalize())
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Box plots side by side
        data_to_plot = [df[giver_col].dropna(), df[guesser_col].dropna()]
        bp = axes[1, 1].boxplot(data_to_plot, labels=['Giver', 'Guesser'], 
                               patch_artist=True)
        for patch, color in zip(bp['boxes'], ['skyblue', 'lightcoral']):
            patch.set_facecolor(color)
        axes[1, 1].set_title(f'{feature_name.capitalize()} - Box Plot Comparison', fontweight='bold')
        axes[1, 1].set_ylabel(feature_name.capitalize())
        
        print(f"\nGiver {feature_name} stats:\n{df[giver_col].describe()}")
        print(f"\nGuesser {feature_name} stats:\n{df[guesser_col].describe()}")
        
    else:
        # Categorical comparison
        giver_counts = df[giver_col].value_counts()
        guesser_counts = df[guesser_col].value_counts()
        
        # Combine for comparison
        all_categories = sorted(set(giver_counts.index) | set(guesser_counts.index))
        
        comparison_df = pd.DataFrame({
            'Giver': [giver_counts.get(cat, 0) for cat in all_categories],
            'Guesser': [guesser_counts.get(cat, 0) for cat in all_categories]
        }, index=all_categories)
        
        print(f"\nCategory comparison:\n{comparison_df}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Bar chart for giver
        giver_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'Giver {feature_name.capitalize()} Distribution', fontweight='bold')
        axes[0, 0].set_xlabel(feature_name.capitalize())
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Bar chart for guesser
        guesser_counts.plot(kind='bar', ax=axes[0, 1], color='lightcoral', edgecolor='black')
        axes[0, 1].set_title(f'Guesser {feature_name.capitalize()} Distribution', fontweight='bold')
        axes[0, 1].set_xlabel(feature_name.capitalize())
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Grouped bar chart
        comparison_df.plot(kind='bar', ax=axes[1, 0], color=['skyblue', 'lightcoral'], 
                          edgecolor='black', width=0.8)
        axes[1, 0].set_title(f'{feature_name.capitalize()} - Side-by-Side Comparison', fontweight='bold')
        axes[1, 0].set_xlabel(feature_name.capitalize())
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        
        # Pie charts
        axes[1, 1].axis('off')
        ax_pie1 = plt.subplot(2, 4, 7)
        ax_pie2 = plt.subplot(2, 4, 8)
        
        giver_counts.plot(kind='pie', ax=ax_pie1, autopct='%1.1f%%', 
                         colors=sns.color_palette('pastel', len(giver_counts)))
        ax_pie1.set_title('Giver Distribution', fontweight='bold')
        ax_pie1.set_ylabel('')
        
        guesser_counts.plot(kind='pie', ax=ax_pie2, autopct='%1.1f%%',
                           colors=sns.color_palette('pastel', len(guesser_counts)))
        ax_pie2.set_title('Guesser Distribution', fontweight='bold')
        ax_pie2.set_ylabel('')
    
    plt.suptitle(f'{dataset_name} - {feature_name.capitalize()} Comparison: Giver vs Guesser', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()

def main():
    """Main analysis function."""
    print("="*80)
    print("CODENAMES LLM INTERPRETABILITY - DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load data
    gg_df, cg_df = load_data()
    
    # Define features to analyze
    categorical_features = ['marriage', 'education', 'race', 'continent', 'language', 
                          'religion', 'gender', 'country', 'political']
    numeric_features = ['age', 'care', 'fairness', 'loyalty', 'authority', 'sanctity',
                       'conscientiousness', 'extraversion', 'neuroticism', 'openness', 
                       'agreeableness']
    
    # Analyze Generate Guess dataset
    print("\n" + "="*80)
    print("ANALYZING GENERATE_GUESS DATASET")
    print("="*80)
    
    for feature in categorical_features:
        for role in ['giver', 'guesser']:
            save_path = output_dir / f'gg_{role}_{feature}_dist.png'
            analyze_categorical_feature(gg_df, feature, role, 'Generate Guess', save_path)
    
    for feature in numeric_features:
        for role in ['giver', 'guesser']:
            save_path = output_dir / f'gg_{role}_{feature}_dist.png'
            analyze_numeric_feature(gg_df, feature, role, 'Generate Guess', save_path)
    
    # Compare roles for generate_guess
    print("\n" + "="*80)
    print("COMPARING GIVER vs GUESSER - GENERATE_GUESS")
    print("="*80)
    
    for feature in categorical_features + numeric_features:
        save_path = output_dir / f'gg_comparison_{feature}.png'
        compare_roles(gg_df, feature, 'Generate Guess', save_path)
    
    # Analyze Clue Generation dataset
    print("\n" + "="*80)
    print("ANALYZING CLUE_GENERATION DATASET")
    print("="*80)
    
    for feature in categorical_features:
        for role in ['giver', 'guesser']:
            save_path = output_dir / f'cg_{role}_{feature}_dist.png'
            analyze_categorical_feature(cg_df, feature, role, 'Clue Generation', save_path)
    
    for feature in numeric_features:
        for role in ['giver', 'guesser']:
            save_path = output_dir / f'cg_{role}_{feature}_dist.png'
            analyze_numeric_feature(cg_df, feature, role, 'Clue Generation', save_path)
    
    # Compare roles for clue_generation
    print("\n" + "="*80)
    print("COMPARING GIVER vs GUESSER - CLUE_GENERATION")
    print("="*80)
    
    for feature in categorical_features + numeric_features:
        save_path = output_dir / f'cg_comparison_{feature}.png'
        compare_roles(cg_df, feature, 'Clue Generation', save_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
