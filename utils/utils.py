# Author: Bora Boyacıoğlu
# Student ID: 150200310

import csv
import re
import time

from collections import defaultdict
from typing import List, Tuple

import spacy

from dataset import DiacritizationDataset

# Load the Turkish language model.
nlp = spacy.blank('tr')

# Define the normalization and tokenization function.
nt = lambda x: tokenize_text(normalize_str(x))

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

def get_scores(predicted: List[List[str]], original_path: str) -> Tuple[float, float]:
    """ Get the scores for the predicted and original texts.
    
    Args:
        predicted (List[str]): The list of predicted texts.
        original_path (str): The path to the original texts.
        
    Returns:
        word_score (float): The word-level accuracy.
        sentence_score (float): The sentence-level accuracy.
    """
    
    # Open the original data.
    with open(original_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Read the sentences by normalising and tokenizing them.
        original = [nt(row[1]) for row in reader][1:-1]
    
    length = (
        len(original),  # The number of sentences.
        sum([len(sentence) for sentence in original])  # The number of words.
    )
        
    word_score = 0
    sentence_score = 0
    
    for i, sent in enumerate(original):
        pred = predicted[i].split(' ')
        
        # Calculate the sentence score.
        if set(pred).issubset(sent):
            sentence_score += 1
        
        # Calculate the word score.
        for word in sent:
            if word in pred:
                word_score += 1
    
    # Normalize the scores.
    word_score /= length[1]
    sentence_score /= length[0]
    
    print(f"\033[1mWord Score:\033[0m\t{100 * word_score:.2f}%\n\033[1mSentence Score:\033[0m\t{100 * sentence_score:.2f}%")
    
    return word_score, sentence_score