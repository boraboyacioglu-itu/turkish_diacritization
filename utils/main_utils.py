import re
import spacy
import time
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
        print(f'\rTokenizing... {100 * (i + 1) / length:.2f}%', end='')
        
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

def verbose_batch(i, time_init, length, skipped, epoch_loss):
    """ Verbose the batch. """

    # Build the progress bar.
    percentage = 100 * (i + 1) / length
    bar = f'[{"=" * int(percentage / 5)}{" " * (20 - int(percentage / 5))}]'
    
    time_diff = time.time() - time_init
    min0, sec0 = divmod(time_diff, 60)
    processed = i + 1 - skipped
    avg_speed = time_diff / processed if processed != 0 else 0
    min1, sec1 = divmod(avg_speed * (length - (i + 1)), 60)

    # Print the statistics.
    print(f"\rBatch {i+1}/{length} {bar} ({percentage:.2f}%)",
          f"Epoch Loss: {epoch_loss / processed if processed != 0 else 0:.4f}",
          f"Skipped: {skipped}",
          f"Elapsed: {int(min0):02d}:{int(sec0):02d}",
          f"Speed: {avg_speed:.2f}s/batch",
          f"Remaining: {int(min1):02d}:{int(sec1):02d}",
          sep=", ", end="")

def untokenize(dataset, index, type, detailed=False):
    # Get the tokens and the vocabulary.
    tokens = dataset.get(index, type)
    vocab = dataset.vocab
    
    sentence = []
    for word in tokens:
        
        # Convert the tokens into words.
        converted = vocab['i2w'][word]
        
        # Break if the padding token is reached.
        if converted == '<pad>':
            break
        
        # Append the converted string.
        sentence.append(str(converted))
    
    # Join the sentence.
    output = None
    if not detailed:
        output = ' '.join(sentence[1:-1])
    else:
        output = ' '.join(sentence) + f' <pad> ({tokens.count(vocab["w2i"]["<pad>"])})...'
    
    return output