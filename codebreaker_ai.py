import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Define cache directory
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'codenames_ai')
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, 'word2vec-google-news-300.pkl')
NLTK_CACHE_PATH = os.path.join(CACHE_DIR, 'nltk_data.pkl')

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def download_and_cache_nltk_data():
    """
    Download necessary NLTK data and cache it.
    """
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def load_nltk_data():
    """
    Load NLTK data from cache or download if not available.
    """
    if not os.path.exists(NLTK_CACHE_PATH):
        download_and_cache_nltk_data()
        with open(NLTK_CACHE_PATH, 'wb') as f:
            pickle.dump(nltk.data._index, f)
    else:
        with open(NLTK_CACHE_PATH, 'rb') as f:
            nltk.data._index = pickle.load(f)

def load_model():
    """
    Load the pre-trained word embedding model from cache or download if not available.
    
    Returns:
    - model: The pre-trained word embedding model.
    """
    if os.path.exists(MODEL_CACHE_PATH):
        with open(MODEL_CACHE_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        model = api.load("word2vec-google-news-300")
        with open(MODEL_CACHE_PATH, 'wb') as f:
            pickle.dump(model, f)
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

def generate_clue(board_words, team_words, opponent_words, bystanders, assassin, model):
    """
    Generate a clue for the Codenames game.

    Parameters:
    - board_words: List of all words on the board.
    - team_words: List of words that belong to the AI's team.
    - opponent_words: List of words that belong to the opponent's team.
    - bystanders: List of words that are neutral.
    - assassin: List containing the assassin word.
    - model: Pre-trained word embedding model.

    Returns:
    - best_clue: The best clue word.
    - candidate_clues[best_clue]: The number of team words the clue relates to.
    """
    candidate_clues = {}
    lemmatized_board_words = {lemmatize_word(word) for word in board_words}

    # Generate candidate clues based on similarity to board words
    for word in board_words:
        if word in model:
            similar_words = model.most_similar(word, topn=50)
            for similar_word, _ in similar_words:
                lemmatized_similar_word = lemmatize_word(similar_word)
                if lemmatized_similar_word not in lemmatized_board_words:
                    candidate_clues[similar_word] = candidate_clues.get(similar_word, 0) + 1

    best_clue = None
    best_score = float('-inf')

    # Evaluate each candidate clue
    for clue, _ in candidate_clues.items():
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

            # Update best clue if the current clue has a higher score
            if score > best_score:
                best_score = score
                best_clue = clue

    return best_clue, candidate_clues[best_clue]

# Load NLTK data and model
load_nltk_data()
model = load_model()

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Prompt the player for input
team_words = get_words("Enter team words separated by spaces: ")
opponent_words = get_words("Enter opponent words separated by spaces: ")
bystanders = get_words("Enter bystanders words separated by spaces: ")
assassin = get_words("Enter the assassin word: ").split()

# Combine all words to form the board words
board_words = team_words + opponent_words + bystanders + assassin

# Generate a clue for the game
clue, number = generate_clue(board_words, team_words, opponent_words, bystanders, assassin, model)
print(f"Clue: {clue}, Number: {number}")