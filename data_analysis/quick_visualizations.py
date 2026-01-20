import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_overview_dashboard(df, dataset_name, save_path=None):
    """Create a comprehensive overview dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Giver Marriage
    ax1 = fig.add_subplot(gs[0, 0])
    df['giver.marriage'].value_counts().plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Giver Marriage Status', fontweight='bold')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)
    
    # Giver Education
    ax2 = fig.add_subplot(gs[0, 1])
    df['giver.education'].value_counts().plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
    ax2.set_title('Giver Education', fontweight='bold')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=45)
    
    # Giver Race
    ax3 = fig.add_subplot(gs[0, 2])
    df['giver.race'].value_counts().plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='black')
    ax3.set_title('Giver Race', fontweight='bold')
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', rotation=45)
    
    # Giver Gender
    ax4 = fig.add_subplot(gs[0, 3])
    df['giver.gender'].value_counts().head(3).plot(kind='bar', ax=ax4, color='plum', edgecolor='black')
    ax4.set_title('Giver Gender', fontweight='bold')
    ax4.set_xlabel('')
    ax4.tick_params(axis='x', rotation=45)
    
    # Giver Age Distribution
    ax5 = fig.add_subplot(gs[1, 0])
    df['giver.age'].dropna().hist(bins=30, ax=ax5, color='skyblue', edgecolor='black')
    ax5.set_title('Giver Age Distribution', fontweight='bold')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Frequency')
    
    # Guesser Age Distribution
    ax6 = fig.add_subplot(gs[1, 1])
    df['guesser.age'].dropna().hist(bins=30, ax=ax6, color='lightcoral', edgecolor='black')
    ax6.set_title('Guesser Age Distribution', fontweight='bold')
    ax6.set_xlabel('Age')
    ax6.set_ylabel('Frequency')
    
    # Giver Political Leaning
    ax7 = fig.add_subplot(gs[1, 2])
    df['giver.political'].value_counts().plot(kind='bar', ax=ax7, color='gold', edgecolor='black')
    ax7.set_title('Giver Political Leaning', fontweight='bold')
    ax7.set_xlabel('')
    ax7.tick_params(axis='x', rotation=45)
    
    # Guesser Political Leaning
    ax8 = fig.add_subplot(gs[1, 3])
    df['guesser.political'].value_counts().plot(kind='bar', ax=ax8, color='orange', edgecolor='black')
    ax8.set_title('Guesser Political Leaning', fontweight='bold')
    ax8.set_xlabel('')
    ax8.tick_params(axis='x', rotation=45)
    
    # Country Distribution
    ax9 = fig.add_subplot(gs[2, 0])
    df['giver.country'].value_counts().head(10).plot(kind='barh', ax=ax9, color='teal', edgecolor='black')
    ax9.set_title('Top 10 Giver Countries', fontweight='bold')
    ax9.set_xlabel('Count')
    
    # Religion Distribution
    ax10 = fig.add_subplot(gs[2, 1])
    df['giver.religion'].value_counts().plot(kind='bar', ax=ax10, color='pink', edgecolor='black')
    ax10.set_title('Giver Religion', fontweight='bold')
    ax10.set_xlabel('')
    ax10.tick_params(axis='x', rotation=45)
    
    # Personality traits comparison (Giver)
    ax11 = fig.add_subplot(gs[2, 2])
    personality_cols = ['giver.care', 'giver.fairness', 'giver.loyalty', 
                       'giver.authority', 'giver.sanctity']
    means = [df[col].mean() for col in personality_cols]
    labels = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Sanctity']
    ax11.bar(labels, means, color='lightblue', edgecolor='black')
    ax11.set_title('Giver Moral Foundations (Mean)', fontweight='bold')
    ax11.set_ylabel('Mean Score')
    ax11.tick_params(axis='x', rotation=45)
    ax11.set_ylim([0, 5])
    
    # Big Five traits comparison (Giver)
    ax12 = fig.add_subplot(gs[2, 3])
    big_five_cols = ['giver.conscientiousness', 'giver.extraversion', 
                     'giver.neuroticism', 'giver.openness', 'giver.agreeableness']
    means = [df[col].mean() for col in big_five_cols]
    labels = ['Conscient.', 'Extraver.', 'Neurot.', 'Openness', 'Agreeable.']
    ax12.bar(labels, means, color='lightgreen', edgecolor='black')
    ax12.set_title('Giver Big Five (Mean)', fontweight='bold')
    ax12.set_ylabel('Mean Score')
    ax12.tick_params(axis='x', rotation=45)
    ax12.set_ylim([0, 5])
    
    plt.suptitle(f'{dataset_name} - Overview Dashboard', fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overview dashboard to: {save_path}")
    
    plt.close()

def create_personality_comparison(df, dataset_name, save_path=None):
    """Create personality and moral foundations comparison between giver and guesser."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Moral Foundations - Giver
    moral_cols_giver = ['giver.care', 'giver.fairness', 'giver.loyalty', 
                        'giver.authority', 'giver.sanctity']
    moral_labels = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Sanctity']
    giver_means = [df[col].mean() for col in moral_cols_giver]
    
    axes[0, 0].bar(moral_labels, giver_means, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Giver Moral Foundations', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Mean Score', fontsize=12)
    axes[0, 0].set_ylim([0, 5])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Moral Foundations - Guesser
    moral_cols_guesser = ['guesser.care', 'guesser.fairness', 'guesser.loyalty', 
                          'guesser.authority', 'guesser.sanctity']
    guesser_means = [df[col].mean() for col in moral_cols_guesser]
    
    axes[0, 1].bar(moral_labels, guesser_means, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Guesser Moral Foundations', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Mean Score', fontsize=12)
    axes[0, 1].set_ylim([0, 5])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Big Five - Giver
    big_five_cols_giver = ['giver.conscientiousness', 'giver.extraversion', 
                           'giver.neuroticism', 'giver.openness', 'giver.agreeableness']
    big_five_labels = ['Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness', 'Agreeableness']
    giver_bf_means = [df[col].mean() for col in big_five_cols_giver]
    
    axes[1, 0].bar(big_five_labels, giver_bf_means, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Giver Big Five Personality Traits', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('Mean Score', fontsize=12)
    axes[1, 0].set_ylim([0, 5])
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Big Five - Guesser
    big_five_cols_guesser = ['guesser.conscientiousness', 'guesser.extraversion', 
                             'guesser.neuroticism', 'guesser.openness', 'guesser.agreeableness']
    guesser_bf_means = [df[col].mean() for col in big_five_cols_guesser]
    
    axes[1, 1].bar(big_five_labels, guesser_bf_means, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Guesser Big Five Personality Traits', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Mean Score', fontsize=12)
    axes[1, 1].set_ylim([0, 5])
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - Personality & Moral Foundations Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved personality comparison to: {save_path}")
    
    plt.close()

def main():
    """Main function - create key visualizations."""
    print("="*80)
    print("CODENAMES LLM INTERPRETABILITY - DATA VISUALIZATIONS")
    print("="*80)
    
    # Load data
    gg_df, cg_df = load_data()
    
    print("\nCreating overview dashboards...")
    create_overview_dashboard(gg_df, 'Generate Guess Dataset', 
                             output_dir / 'gg_overview_dashboard.png')
    create_overview_dashboard(cg_df, 'Clue Generation Dataset', 
                             output_dir / 'cg_overview_dashboard.png')
    
    print("\nCreating personality comparisons...")
    create_personality_comparison(gg_df, 'Generate Guess Dataset',
                                 output_dir / 'gg_personality_comparison.png')
    create_personality_comparison(cg_df, 'Clue Generation Dataset',
                                 output_dir / 'cg_personality_comparison.png')
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
