# Codenames Dataset Analysis & Preprocessing Guide

## Dataset Overview

You have 3 main CSV files from Codenames game experiments studying LLM behavior with different demographic/personality profiles:

### Files Summary
- **clue_generation.csv** (18.5MB, 7,704 rows): Spymaster giving clues
- **correct_guess.csv** (28.5MB, 9,550 rows): Validating if guesses were correct
- **generate_guess.csv** (20.7MB, 7,704 rows): Guesser making word selections

---

## Dataset Structure

### Common Schema (All 3 Files)

Each file has **8 columns**:

1. **Index column** (`Unnamed: 0` or `index`): Row identifier
2. **base_text**: Core game state without player demographics
3. **leaning_only**: Game state + political/moral leanings
4. **event_only**: Game state + event context
5. **demo_only**: Game state + demographics only
6. **personality_only**: Game state + Big Five personality traits
7. **all_text**: Complete context (all demographic/personality info)
8. **output**: The result/answer

### Game Elements in Text Columns

All text columns contain structured game information:
- **GIVER** demographics (spymaster giving clues)
- **GUESSER** demographics (player making guesses)
- **black**: Assassin words (avoid these!)
- **tan**: Neutral/opponent words
- **targets**: Words the team needs to guess
- **remaining**: Words still on the board
- **hint**: The clue word given
- **rationale**: Reasoning for choices

### Demographic Features

Player profiles can include:
- **Demographics**: marriage, education, race, continent, language, religion, age, gender, country, native status
- **Moral Foundations**: care, fairness, loyalty, authority, sanctity
- **Political leaning**: liberal, conservative, moderate, etc.
- **Personality (Big Five)**: conscientiousness, extraversion, neuroticism, openness, agreeableness

---

## File-Specific Details

### 1. clue_generation.csv
**Task**: Spymaster generates a clue word to help guesser find target words

**Input columns contain**:
- Board state (black/tan/target words)
- Player demographics in various combinations

**Output**: Single word clue (e.g., "sandwich", "baseball", "investment")

**Example**:
```
targets: ['pitch', 'field']
→ output: "baseball"
```

### 2. correct_guess.csv
**Task**: Validate if a guess matches the intended target

**Input columns contain**:
- Remaining words on board
- Hint given
- Target word
- Rationale for the guess

**Output**: Boolean (True/False) - whether guess was correct

**Example**:
```
hint: baseball
target: pitch
rationale: baseball involves pitching
→ output: True
```

### 3. generate_guess.csv
**Task**: Guesser selects word(s) based on hint

**Input columns contain**:
- Remaining words
- Hint provided by spymaster

**Output**: Word(s) guessed (single word or comma-separated)

**Example**:
```
hint: baseball
→ output: "pitch, lap"
```

---

## Preprocessing Recommendations

### 1. Data Cleaning

```python
import pandas as pd
import ast

# Load data
df_clue = pd.read_csv('data/clue_generation.csv')
df_guess = pd.read_csv('data/generate_guess.csv')
df_correct = pd.read_csv('data/correct_guess.csv')

# Drop redundant index columns
df_clue = df_clue.drop(columns=['Unnamed: 0'])
df_correct = df_correct.drop(columns=['index'])
df_guess = df_guess.drop(columns=['Unnamed: 0'])

# Reset index
df_clue.reset_index(drop=True, inplace=True)
df_correct.reset_index(drop=True, inplace=True)
df_guess.reset_index(drop=True, inplace=True)
```

### 2. Parse Structured Text Fields

The text columns contain structured information that should be parsed:

```python
import re

def parse_player_features(text):
    """Extract GIVER and GUESSER demographics from text"""
    features = {}
    
    # Extract GIVER attributes
    giver_match = re.search(r'GIVER: \[(.*?)\]', text)
    if giver_match:
        giver_str = giver_match.group(1)
        features['giver'] = parse_attributes(giver_str)
    
    # Extract GUESSER attributes
    guesser_match = re.search(r'GUESSER: \[(.*?)\]', text)
    if guesser_match:
        guesser_str = guesser_match.group(1)
        features['guesser'] = parse_attributes(guesser_str)
    
    return features

def parse_attributes(attr_str):
    """Parse key: value pairs"""
    attrs = {}
    pairs = attr_str.split(', ')
    for pair in pairs:
        if ': ' in pair:
            key, val = pair.split(': ', 1)
            try:
                attrs[key] = float(val) if '.' in val else val
            except:
                attrs[key] = val
    return attrs

def parse_game_state(text):
    """Extract board state (words)"""
    state = {}
    
    # Extract word lists
    for field in ['black', 'tan', 'targets', 'remaining']:
        match = re.search(rf"{field}: (\[.*?\])", text)
        if match:
            try:
                state[field] = ast.literal_eval(match.group(1))
            except:
                pass
    
    # Extract hint
    hint_match = re.search(r'hint: (\w+)', text)
    if hint_match:
        state['hint'] = hint_match.group(1)
    
    return state
```

### 3. Feature Extraction Strategy

```python
def extract_features(df, text_column='all_text'):
    """Extract structured features from text column"""
    
    # Parse all rows
    parsed_data = []
    for idx, row in df.iterrows():
        text = row[text_column]
        
        features = {}
        features.update(parse_player_features(text))
        features.update(parse_game_state(text))
        features['output'] = row['output']
        
        parsed_data.append(features)
    
    return pd.DataFrame(parsed_data)

# Apply to datasets
clue_features = extract_features(df_clue)
guess_features = extract_features(df_guess)
correct_features = extract_features(df_correct)
```

### 4. Create Feature Matrices

For machine learning, separate demographics from game state:

```python
def create_demographic_features(parsed_df):
    """Flatten demographic dictionaries to columns"""
    
    # Extract giver features
    giver_df = pd.json_normalize(parsed_df['giver'])
    giver_df.columns = ['giver_' + col for col in giver_df.columns]
    
    # Extract guesser features
    guesser_df = pd.json_normalize(parsed_df['guesser'])
    guesser_df.columns = ['guesser_' + col for col in guesser_df.columns]
    
    return pd.concat([giver_df, guesser_df], axis=1)

def encode_word_lists(parsed_df):
    """Convert word lists to useful features"""
    features = {}
    
    features['n_black'] = parsed_df['black'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    features['n_tan'] = parsed_df['tan'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    features['n_targets'] = parsed_df['targets'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    features['n_remaining'] = parsed_df['remaining'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return pd.DataFrame(features)
```

### 5. Handle Different Context Conditions

Your data has 6 context variations for ablation studies:

```python
context_columns = [
    'base_text',         # No demographics
    'leaning_only',      # Political/moral only
    'event_only',        # Event context only
    'demo_only',         # Demographics only
    'personality_only',  # Big Five only
    'all_text'          # All information
]

# Create separate datasets for each condition
datasets = {}
for col in context_columns:
    datasets[col] = extract_features(df_clue, text_column=col)
```

---

## Analysis Suggestions

### 1. Performance Analysis
- Compare output quality across different context conditions
- Measure correctness rates from `correct_guess.csv`
- Analyze which demographic features correlate with better/worse performance

### 2. Bias Detection
- Test if certain demographic profiles lead to different clue strategies
- Check for systematic differences by gender, race, political leaning
- Measure fairness across demographic groups

### 3. Interpretability Studies
- Which demographic features most influence clue generation?
- Do personality traits predict guessing accuracy?
- Feature importance analysis for each context condition

### 4. Temporal Analysis
- Track how game state evolves (words remaining)
- Success rate as game progresses
- Learning effects across rounds

---

## Quick Start Code

```python
# Complete preprocessing pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df_clue = pd.read_csv('data/clue_generation.csv').drop(columns=['Unnamed: 0'])
df_guess = pd.read_csv('data/generate_guess.csv').drop(columns=['Unnamed: 0'])
df_correct = pd.read_csv('data/correct_guess.csv').drop(columns=['index'])

# 2. Extract features (use functions above)
clue_features = extract_features(df_clue)

# 3. Create demographics matrix
demo_features = create_demographic_features(clue_features)

# 4. Create game state features
game_features = encode_word_lists(clue_features)

# 5. Combine for modeling
X = pd.concat([demo_features, game_features], axis=1)
y = clue_features['output']

# 6. Handle categorical variables
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 7. Fill missing values
X.fillna(X.median(), inplace=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

---

## Key Research Questions

Based on this data structure, you can investigate:

1. **Demographic Impact**: Do player demographics affect clue quality/guessing?
2. **Context Ablation**: Which context (base/leaning/demo/personality/all) works best?
3. **Fairness**: Are certain groups disadvantaged by demographic conditioning?
4. **Interpretability**: What features drive LLM decisions?
5. **Generalization**: Does conditioning help or hurt performance?

---

## Next Steps

1. Run the parsing functions to extract structured features
2. Explore each context condition separately
3. Build baseline models on `base_text` only
4. Compare with demographic-conditioned versions
5. Analyze feature importance and bias patterns
