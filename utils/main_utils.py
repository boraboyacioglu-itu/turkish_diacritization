import re
import spacy

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