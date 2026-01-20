import os
import requests

# Define the target folder
TARGET_FOLDER = "data"

# Ensure the target folder exists
os.makedirs(TARGET_FOLDER, exist_ok=True)

# List of URLs and their corresponding filenames
files_to_download = {
    "https://raw.githubusercontent.com/SALT-NLP/codenames/main/data/clue_generation_task/all.csv": "clue_generation.csv",
    "https://raw.githubusercontent.com/SALT-NLP/codenames/main/data/correct_guess_task/all.csv": "correct_guess.csv",
    "https://raw.githubusercontent.com/SALT-NLP/codenames/main/data/generate_guess_task/all.csv": "generate_guess.csv",
    "https://raw.githubusercontent.com/SALT-NLP/codenames/main/data/guess_rationale_task/all.csv": "guess_rationale.csv",
    "https://raw.githubusercontent.com/SALT-NLP/codenames/main/data/target_rationale_task/all.csv": "target_rationale.csv",
    "https://raw.githubusercontent.com/SALT-NLP/codenames/main/data/target_selection_task/all.csv": "target_selection.csv",
}

# Function to download a file
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(TARGET_FOLDER, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download {filename}. HTTP Status Code: {response.status_code}")

# Download each file
for url, filename in files_to_download.items():
    download_file(url, filename)