# Data Analysis Quick Start Guide

## Installation

Ensure you have the required packages:
```bash
pip3 install pandas matplotlib seaborn numpy
```

## Quick Commands

### 1. Generate Summary Statistics (Fast - ~5 seconds)
```bash
python3 data_analysis/summary_statistics.py
```
**Output:** Console output + `output/data_analysis/summary_statistics.txt`

### 2. Create Overview Dashboards (Fast - ~10 seconds)
```bash
python3 data_analysis/quick_visualizations.py
```
**Output:** 4 PNG files in `output/data_analysis/`
- `gg_overview_dashboard.png` - 12-panel overview of Generate Guess data
- `cg_overview_dashboard.png` - 12-panel overview of Clue Generation data  
- `gg_personality_comparison.png` - Personality traits comparison
- `cg_personality_comparison.png` - Personality traits comparison

### 3. Explore Specific Features (Interactive)
```bash
# Explore age for givers in both datasets
python3 data_analysis/explore.py age giver

# Explore education for guessers in Generate Guess dataset only
python3 data_analysis/explore.py education guesser gg

# Explore political leaning for givers in Clue Generation dataset
python3 data_analysis/explore.py political giver cg

# Show available features
python3 data_analysis/explore.py --list
```

### 4. Detailed Feature Analysis (Slow - ~5-10 minutes)
```bash
python3 data_analysis/feature_distributions.py
```
**Output:** 100+ PNG files with detailed distributions and comparisons

---

## Example Workflow

**Quick exploration (< 1 minute):**
```bash
# Get statistics
python3 data_analysis/summary_statistics.py | head -100

# Create overview charts
python3 data_analysis/quick_visualizations.py

# Explore specific features
python3 data_analysis/explore.py age
python3 data_analysis/explore.py education giver
python3 data_analysis/explore.py care guesser
```

**Deep dive (several minutes):**
```bash
# Generate all detailed visualizations
python3 data_analysis/feature_distributions.py
```

---

## Understanding the Output

### Summary Statistics
- **Categorical features:** Shows top 5 values with counts and percentages
- **Numeric features:** Shows mean, std, min, 25%, median, 75%, max
- **Missing values:** Tracked for all features
- **Comparison:** Side-by-side GIVER vs GUESSER stats

### Overview Dashboard (12 panels)
1. Giver Marriage Status
2. Giver Education Level  
3. Giver Race Distribution
4. Giver Gender Distribution
5. Giver Age Histogram
6. Guesser Age Histogram
7. Giver Political Leaning
8. Guesser Political Leaning
9. Top 10 Giver Countries
10. Giver Religion Distribution
11. Moral Foundations Mean Scores
12. Big Five Traits Mean Scores

### Personality Comparison (4 panels)
1. Giver Moral Foundations
2. Guesser Moral Foundations
3. Giver Big Five Traits
4. Guesser Big Five Traits

---

## Tips

1. **Start small:** Run `summary_statistics.py` and `quick_visualizations.py` first
2. **Use explore.py:** Great for quick checks without generating files
3. **Save detailed analysis for last:** `feature_distributions.py` generates many files
4. **Check output folder:** All visualizations save to `output/data_analysis/`
5. **Combine datasets:** Most commands analyze both datasets by default

---

## Common Questions

**Q: How many records are in each dataset?**
A: Both have 7,703 rows

**Q: What's the difference between the datasets?**
A: Both have same GIVER/GUESSER demographics but different task-specific columns:
- Generate Guess: `remaining`, `hint`, `output`
- Clue Generation: `black`, `tan`, `targets`, `output`

**Q: Why are there missing values?**
A: Some participants didn't provide all demographic information

**Q: What do the moral foundations measure?**
A: Five dimensions from Moral Foundations Theory:
- Care/Harm
- Fairness/Cheating
- Loyalty/Betrayal
- Authority/Subversion
- Sanctity/Degradation

**Q: What are the Big Five traits?**
A: Five major personality dimensions:
- Conscientiousness
- Extraversion
- Neuroticism
- Openness
- Agreeableness

---

## Next Steps

After running the analysis scripts, you can:
1. Review `output/data_analysis/summary_statistics.txt` for key insights
2. Open the PNG files to visualize distributions
3. Use findings to inform further research questions
4. Build models based on the demographic and personality features

For more details, see `data_analysis/README.md`
