# Data Analysis Scripts

This folder contains scripts for analyzing the preprocessed Codenames LLM Interpretability datasets.

## Overview

The data analysis pipeline helps understand the distribution of demographic, personality, and moral foundation features across both the **Generate Guess** and **Clue Generation** datasets.

## Available Scripts

### 1. `summary_statistics.py`
Generates comprehensive statistical summaries for both datasets.

**Features:**
- Categorical feature distributions (marriage, education, race, continent, language, religion, gender, country, political leaning)
- Numeric feature statistics (age, moral foundations, Big Five personality traits)
- Comparison between GIVER and GUESSER roles
- Outputs results to console and saves to `output/data_analysis/summary_statistics.txt`

**Usage:**
```bash
python3 data_analysis/summary_statistics.py
```

**Output:**
- Console output with detailed statistics
- `output/data_analysis/summary_statistics.txt` - Full summary report

---

### 2. `quick_visualizations.py`
Creates high-level overview dashboards and personality comparisons.

**Features:**
- Overview dashboard with 12 key visualizations per dataset
  - Demographic distributions (marriage, education, race, gender)
  - Age distributions for both GIVER and GUESSER
  - Political leaning distributions
  - Top 10 countries
  - Religion distribution
  - Moral foundations mean scores
  - Big Five personality traits mean scores
- Personality comparison charts
  - Moral foundations (care, fairness, loyalty, authority, sanctity)
  - Big Five traits (conscientiousness, extraversion, neuroticism, openness, agreeableness)

**Usage:**
```bash
python3 data_analysis/quick_visualizations.py
```

**Output:**
- `output/data_analysis/gg_overview_dashboard.png` - Generate Guess overview
- `output/data_analysis/cg_overview_dashboard.png` - Clue Generation overview
- `output/data_analysis/gg_personality_comparison.png` - Generate Guess personality
- `output/data_analysis/cg_personality_comparison.png` - Clue Generation personality

---

### 3. `feature_distributions.py`
Comprehensive feature-by-feature analysis with detailed visualizations.

**Features:**
- Individual analysis for each categorical feature (bar charts, pie charts)
- Individual analysis for each numeric feature (histograms, box plots)
- Role comparisons (GIVER vs GUESSER) for all features
- Generates 100+ visualization files for deep-dive analysis

**Usage:**
```bash
python3 data_analysis/feature_distributions.py
```

**Output:**
Multiple PNG files in `output/data_analysis/`:
- `{dataset}_{role}_{feature}_dist.png` - Individual feature distributions
- `{dataset}_comparison_{feature}.png` - GIVER vs GUESSER comparisons

**Note:** This script generates many files and takes longer to run. Use for detailed feature exploration.

---

## Data Structure

### Datasets Analyzed
Both scripts analyze two preprocessed datasets:
- `data/preprocessed/generate_guess.csv` (7,703 rows × 45 columns)
- `data/preprocessed/clue_generation.csv` (7,703 rows × 46 columns)

### Features Analyzed

#### Categorical Features
- **Demographic:** marriage, education, race, continent, language, religion, gender, country
- **Psychological:** political (leaning)

#### Numeric Features
- **Age:** Continuous variable
- **Moral Foundations Theory (5 dimensions):**
  - care
  - fairness
  - loyalty
  - authority
  - sanctity
- **Big Five Personality Traits (5 dimensions):**
  - conscientiousness
  - extraversion
  - neuroticism
  - openness
  - agreeableness

### Roles
Each dataset contains data for two roles:
- **GIVER:** The player giving clues
- **GUESSER:** The player receiving and interpreting clues

---

## Requirements

```bash
pip3 install pandas matplotlib seaborn numpy
```

---

## Quick Start

To get started with data analysis:

1. **Generate summary statistics:**
   ```bash
   python3 data_analysis/summary_statistics.py
   ```

2. **Create overview visualizations:**
   ```bash
   python3 data_analysis/quick_visualizations.py
   ```

3. **For detailed analysis (optional):**
   ```bash
   python3 data_analysis/feature_distributions.py
   ```

All outputs will be saved to `output/data_analysis/`.

---

## Output Directory Structure

```
output/data_analysis/
├── summary_statistics.txt              # Statistical summary report
├── gg_overview_dashboard.png          # Generate Guess overview
├── cg_overview_dashboard.png          # Clue Generation overview
├── gg_personality_comparison.png      # Generate Guess personality
├── cg_personality_comparison.png      # Clue Generation personality
└── [Additional files from feature_distributions.py]
```

---

## Key Insights to Explore

The analysis scripts help answer questions like:
- What are the demographic distributions of participants?
- How do GIVER and GUESSER personalities differ?
- What are the most common moral foundations values?
- Are there differences in Big Five traits between roles?
- What is the geographic distribution of participants?
- What are the political leanings of participants?

---

## Notes

- All visualizations use consistent color schemes for easy comparison
- Missing values are tracked and reported in statistics
- The scripts handle data parsing automatically (lists stored as strings are handled correctly)
- High-resolution PNG files (300 DPI) are suitable for publication

---

## Future Enhancements

Potential additions to the analysis pipeline:
- Correlation analysis between features
- Statistical significance testing between GIVER and GUESSER
- Clustering analysis based on personality profiles
- Time-series analysis if temporal data is available
- Interactive dashboards using Plotly or Streamlit
