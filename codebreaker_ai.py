import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle
import os
import logging
from datetime import datetime
from tqdm import tqdm

# Flip this bit to enable testing
TESTING_MODE = True

# Define cache directory
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'codenames_ai')
LOG_DIR = './logs'
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, 'word2vec-google-news-300.pkl')
NLTK_FLAG_PATH = os.path.join(CACHE_DIR, 'nltk_data_downloaded.flag')

# Ensure the cache and log directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_filename = os.path.join(LOG_DIR, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_cache_nltk_data():
    """
    Download necessary NLTK data and create a flag file to indicate completion.
    """
    logging.info("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    with open(NLTK_FLAG_PATH, 'w') as f:
        f.write('')
    logging.info("NLTK data download complete.")

def ensure_nltk_data():
    """
    Ensure NLTK data is downloaded by checking the flag file or downloading if not available.
    """
    if not os.path.exists(NLTK_FLAG_PATH):
        download_and_cache_nltk_data()

def load_model():
    """
    Load the pre-trained word embedding model from cache or download if not available.
    
    Returns:
    - model: The pre-trained word embedding model.
    """
    if os.path.exists(MODEL_CACHE_PATH):
        logging.info("Loading model from cache...")
        with open(MODEL_CACHE_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        logging.info("Downloading model...")
        model = api.load("word2vec-google-news-300")
        with open(MODEL_CACHE_PATH, 'wb') as f:
            pickle.dump(model, f)
        logging.info("Model download complete.")
    return model

def get_words(prompt):
    """
    Prompt the player to input words separated by spaces.
    
    Parameters:
    - prompt: The message to display to the player.
    
    Returns:
    - List of words entered by the player.
    """
    return input(prompt).split()

def lemmatize_word(word):
    """
    Lemmatize a given word to its base form.
    
    Parameters:
    - word: The word to lemmatize.
    
    Returns:
    - Lemmatized word.
    """
    return lemmatizer.lemmatize(word)

def is_valid_clue(clue):
    """
    Check if the clue is a valid single word without hyphens or underscores.
    
    Parameters:
    - clue: The clue word to check.
    
    Returns:
    - bool: True if the clue is valid, False otherwise.
    """
    return len(clue.split()) == 1 and '-' not in clue and '_' not in clue

def generate_clues(board_words, team_words, opponent_words, bystanders, assassin, model, top_n=5):
    """
    Generate a list of potential clues for the Codenames game.

    Parameters:
    - board_words: List of all words on the board.
    - team_words: List of words that belong to the AI's team.
    - opponent_words: List of words that belong to the opponent's team.
    - bystanders: List of words that are neutral.
    - assassin: List containing the assassin word.
    - model: Pre-trained word embedding model.
    - top_n: Number of top clues to return.

    Returns:
    - List of top_n clues with confidence and the number of words.
    """
    logging.info("Generating candidate clues...")
    candidate_clues = {}
    lemmatized_board_words = {lemmatize_word(word.lower()) for word in board_words}

    # Generate candidate clues based on similarity to board words
    for word in tqdm(board_words, desc="Processing board words"):
        if word in model:
            similar_words = model.most_similar(word, topn=50)
            for similar_word, _ in similar_words:
                lemmatized_similar_word = lemmatize_word(similar_word.lower())
                if lemmatized_similar_word not in lemmatized_board_words and is_valid_clue(similar_word):
                    if similar_word.lower() not in candidate_clues:
                        candidate_clues[similar_word.lower()] = {'count': 0, 'words': []}
                    candidate_clues[similar_word.lower()]['count'] += 1
                    candidate_clues[similar_word.lower()]['words'].append(word)

    logging.info("Evaluating candidate clues...")
    scored_clues = []

    # Evaluate each candidate clue
    for clue in tqdm(candidate_clues, desc="Evaluating clues"):
        if clue in model:
            score = 0
            # Calculate similarity score for team words
            for word in team_words:
                if word in model:
                    score += cosine_similarity([model[clue]], [model[word]])[0][0]
            # Subtract similarity score for opponent words
            for word in opponent_words:
                if word in model:
                    score -= cosine_similarity([model[clue]], [model[word]])[0][0]
            # Subtract similarity score for bystanders
            for word in bystanders:
                if word in model:
                    score -= cosine_similarity([model[clue]], [model[word]])[0][0]
            # Subtract similarity score for the assassin
            if assassin[0] in model:
                score -= cosine_similarity([model[clue]], [model[assassin[0]]])[0][0]

            scored_clues.append((clue, score, candidate_clues[clue]['count']))

    # Normalize scores to range [0, 100]
    min_score = min(scored_clues, key=lambda x: x[1])[1]
    max_score = max(scored_clues, key=lambda x: x[1])[1]
    scored_clues = [(clue, (score - min_score) / (max_score - min_score) * 100, num_words) for clue, score, num_words in scored_clues]

    # Sort clues by confidence and return the top_n clues
    scored_clues.sort(key=lambda x: x[1], reverse=True)
    top_clues = scored_clues[:top_n]

    logging.info(f"Top clues generated: {top_clues}")
    return top_clues

def update_board_words(board_words, guessed_words):
    """
    Update the board words by removing the guessed words.

    Parameters:
    - board_words: List of all words on the board.
    - guessed_words: List of words that have been guessed.

    Returns:
    - Updated list of board words.
    """
    return [word for word in board_words if word not in guessed_words]

# Ensure NLTK data is available
ensure_nltk_data()

# Load the word embedding model
model = load_model()

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

if TESTING_MODE:
    # Pre-populate the 
    team_words = "step plastic stable medic craft eagle scorpion bat lab"
    opponent_words = "back horn santa marble genius unicorn mohawk jam"
    bystanders = "czech garden roulette figure slug thumb puppet"
    assassin = "submarine"
else:
    # Prompt the player for input
    team_words = get_words("Enter team words separated by spaces: ")
    opponent_words = get_words("Enter opponent words separated by spaces: ")
    bystanders = get_words("Enter bystanders words separated by spaces: ")
    assassin = get_words("Enter the assassin word: ")

# Combine all words to form the board words
board_words = team_words + opponent_words + bystanders + assassin

# Log the input words
logging.info(f"Team words: {team_words}")
logging.info(f"Opponent words: {opponent_words}")
logging.info(f"Bystanders: {bystanders}")
logging.info(f"Assassin: {assassin}")

while True:
    # Generate a list of potential clues for the game
    top_clues = generate_clues(board_words, team_words, opponent_words, bystanders, assassin, model)
    print("Top clues:")
    for i, (clue, confidence, num_words) in enumerate(top_clues, 1):
        print(f"{i}. {clue} (Confidence: {confidence:.2f}% Words: {num_words})")

    # Prompt the spymaster to choose a clue
    chosen_index = int(input("Choose a clue by entering its number: ")) - 1
    chosen_clue = top_clues[chosen_index][0]
    logging.info(f"Chosen clue: {chosen_clue}")

    # Prompt the user for the words guessed
    guessed_words = get_words("Enter the words guessed separated by spaces: ")
    logging.info(f"Guessed words: {guessed_words}")

    # Update the board words by removing the guessed words
    board_words = update_board_words(board_words, guessed_words)
    logging.info(f"Updated board words: {board_words}")

    # Check if the game is over
    if not team_words or not board_words:
        print("Game over!")
        logging.info("Game over!")
        break