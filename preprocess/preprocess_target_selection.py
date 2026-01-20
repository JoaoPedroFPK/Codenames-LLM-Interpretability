import pandas as pd
import re
import ast

def parse_attributes(text, prefix):
    """Parse GIVER or GUESSER attributes from text."""
    pattern = f"{prefix}: \\[([^\\]]+)\\]"
    match = re.search(pattern, text)
    if not match:
        return {}
    
    attrs_text = match.group(1)
    attrs = {}
    
    # Split by comma, but be careful with values that contain commas
    parts = re.split(r',\s*(?=[a-z_]+:)', attrs_text)
    
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
            
            attrs[f"{prefix.lower()}.{key}"] = value
    
    return attrs

def parse_word_list(text, list_name):
    """Parse green, black, or tan list from text."""
    pattern = f"{list_name}:\\s*(\\[.*?\\])"
    match = re.search(pattern, text)
    if not match:
        return []
    
    try:
        return ast.literal_eval(match.group(1))
    except:
        return []

# Load the raw data
df = pd.read_csv('data/raw/target_selection.csv')

# Create list to hold parsed rows
parsed_rows = []

for idx, row in df.iterrows():
    all_text = row['all_text']
    
    # Parse GIVER and GUESSER attributes
    giver_attrs = parse_attributes(all_text, 'GIVER')
    guesser_attrs = parse_attributes(all_text, 'GUESSER')
    
    # Parse word lists
    green = parse_word_list(all_text, 'green')
    black = parse_word_list(all_text, 'black')
    tan = parse_word_list(all_text, 'tan')
    
    # Combine all parsed data
    parsed_row = {
        **giver_attrs,
        **guesser_attrs,
        'green': green,
        'black': black,
        'tan': tan,
        'output': row['output']
    }
    
    parsed_rows.append(parsed_row)

# Create new dataframe
preprocessed_df = pd.DataFrame(parsed_rows)

# Save to CSV
preprocessed_df.to_csv('data/preprocessed/target_selection.csv', index=False)

print(f"Preprocessed {len(preprocessed_df)} rows")
print(f"\nColumns: {list(preprocessed_df.columns)}")
print(f"\nFirst row sample:")
print(preprocessed_df.iloc[0].to_dict())
