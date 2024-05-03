import re
import spacy
from collections import defaultdict

from dataset import DiacritizationDataset

# Load the Turkish language model.
nlp = spacy.blank('tr')

def normalize_str(text: str) -> str:
    """ Normalize a string by converting it to lowercase and removing non-alphanumeric characters. """
    # Convert the text to lowercase.
    text = text.lower()
    
    # Remove non-alphanumeric characters.
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove multiple spaces.
    text = re.sub(r'  ', ' ', text)
    
    return text

def normalize(texts: DiacritizationDataset):
    """ Apply string normalization to the texts in the dataset. """
    length = len(texts)
    
    for i in range(length):
        # Print the progess.
        print(f'\rNormalizing text {100 * (i + 1) / length:.2f}%', end='')
        
        for type in ['und', 'd']:
            # Get the text.
            text = texts.get(i, type)
            
            # Normalize the text if it exists.
            if text:
                texts.set(i, type, normalize_str(text))
    
    print()

def tokenize_text(text: str) -> list[str]:
    """ Tokenize a text by splitting it into words. """    
    
    # Tokenize the text.
    doc = nlp(text)
    
    # Get the tokens.
    tokenized = [token.text for token in doc]
    
    # Remove empty strings.
    tokenized = [word for word in tokenized if word not in ['', ' ']]
    
    return tokenized

def tokenize(texts: DiacritizationDataset):
    """ Tokenize the texts in the dataset. """
    length = len(texts)
    
    for i in range(length):
        # Print the progess.
        print(f'\rTokenizing text {100 * (i + 1) / length:.2f}%', end='')
        
        for type in ['und', 'd']:
            # Get the text.
            text = texts.get(i, type)
            
            # If it is already tokenized, skip it.
            if isinstance(text, list):
                continue
            
            # Tokenize the text if it exists.
            if text:
                texts.set(i, type, tokenize_text(text))
    
    print()

def build_vocab(texts):
    vocab = defaultdict(int)
    
    # Add special marker tokens to the vocabulary.
    special_tokens = ['<sos>', '<eos>', '<unk>', '<pad>']
    for token in special_tokens:
        vocab[token] = 10**3 * len(texts)  # A large enough number
    
    # Count the frequency of each word in the dataset.
    for text in texts:
        for token in text:
            vocab[token] += 1
    
    # Sort the vocabulary by frequency.
    vocab_list = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    
    # Create the word to index and index to word mappings.
    w2i = {word: idx for idx, (word, _) in enumerate(vocab_list)}
    i2w = {idx: word for word, idx in w2i.items()}

    return w2i, i2w