# Author: Bora Boyacıoğlu
# Student ID: 150200310

import pickle as pkl
import re
import torch

from collections import Counter
from unidecode import unidecode

from torch.utils.data import Dataset

class DiacritizationDataset(Dataset):
    def __init__(self, file_name, type, filter=True):
        # Read the data from the file.
        text: list = self.read_data(file_name, filter)
        
        # Initialize the diacritized and undiacritized data.
        self.diacritized = None
        self.undiacritized = None
        
        # Define the vocab.
        self.vocab = None
        
        self.indiced = False
        
        if not text:
            print('\033[91mNo data found\033[0m')
        elif type == 'train':
            
            # If the data contains original Turkish sentences,
            # unidecode them to remove diacritics.
            self.diacritized = text
            self.undiacritized = self.undiacritize(text)
        elif type == 'test':
                        
            # Use the data as it is for predictions.
            self.undiacritized = text
        else:
            print('\033[91mInvalid type\033[0m')
        
    def read_data(self, file_name, filter):
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                # Read the CSV file.
                data = f.readlines()
                
                # Skip the header and get the sentences.
                text = []
                for line in data[1:]:
                    sentence = line.strip().split(',')
                    try:
                        int(sentence[0])
                        sentence = sentence[1:]
                    except ValueError:
                        None
                    in_str = ''.join(sentence).replace('"', '')
                    
                    # Filter the outliers.
                    new = self.filter_text(in_str, 120) if filter else [in_str]
                    
                    # Append the sentence if it is not empty.
                    text.extend(new) if new else None
                    
                return text
        except FileNotFoundError:
            print(f'\033[91mFile {file_name} not found\033[0m')
            return []
    
    def save_data(self, file_name):
        with open(file_name, 'wb') as f:
            pkl.dump(self, f)
    
    def save_vocab(self, file_name):
        with open(file_name, 'wb') as f:
            pkl.dump(self.vocab, f)
    
    def undiacritize(self, text):
        return [unidecode(sentence) for sentence in text]
    
    def get(self, index, type):
        if type == 'd':
            return self.diacritized[index] if self.diacritized else None
        elif type == 'und':
            return self.undiacritized[index]
        else:
            print('\033[91mInvalid type\033[0m')
            return None
    
    def set(self, index, type, new):
        if type == 'd':
            self.diacritized[index] = new
        elif type == 'und':
            self.undiacritized[index] = new
        else:
            print('\033[91mInvalid type\033[0m')
    
    def filter_text(self, text: str, len_lim: int) -> list[str]:
        splits = re.split(';|/|,', text)
        returns = []
        
        for split in splits:
            words = split.split(' ')
            l = len(words)
            
            if l <= len_lim:
                returns.append(split)
            else:
                returns.extend([
                    ' '.join(words[:l//2]),
                    ' '.join(words[l//2:])
                ])
        return returns
            
    
    def pad(self, max_len=None):
        # Get the maximum length of the sentences.
        max_len = max_len or max([len(sentence) for sentence in self.undiacritized])
        
        # Define the sandwich function for padding.
        sandwich = lambda sentence, max_len: ['<sos>'] + sentence + ['<eos>'] + ['<pad>'] * (max_len - len(sentence))
        
        # Pad the undiacritized texts.
        self.undiacritized = [
            sandwich(sentence, max_len) for sentence in self.undiacritized if type(sentence) == list
        ]
        
        # Skip the diacritized texts if they do not exist.
        if not self.diacritized:
            return max_len
        
        # Pad the diacritized texts.
        self.diacritized = [
            sandwich(sentence, max_len) for sentence in self.diacritized if type(sentence) == list
        ]
        
        return max_len
    
    def to_indices(self, force=False):
        # Skip if the data is already indiced or the vocabulary does not exist.
        if (self.indiced and not force) or not self.vocab:
            return
        
        self.indiced = True
        
        unk = self.vocab['w2i']['<unk>']
        
        # Convert the undiacritized texts to indices using the vocabulary.
        self.undiacritized = [
            [self.vocab['w2i'].get(token, unk) for token in sentence
                ] for sentence in self.undiacritized
        ]
        
        # Skip the diacritized texts if they do not exist.
        if not self.diacritized:
            return
        
        # Convert the diacritized texts to indices using the vocabulary, if they exist.
        self.diacritized = [
            [self.vocab['w2i'].get(token, unk) for token in sentence
                ] for sentence in self.diacritized
        ] if self.diacritized else None
        
    def build_vocab(self, test_data):        
        # Skip if this is not the train data.
        if not self.diacritized:
            print('\033[91mVocabulary can only be built in Train Data.\033[0m')
            return
        
        # Skip if the test data is not an instance of DiacritizationDataset.
        if not isinstance(test_data, DiacritizationDataset):
            print('\033[91mTest data must be an instance of DiacritizationDataset.\033[0m')
            return
        
        # Create the list of all the words
        words = [
            token
            for sentence in (self.undiacritized + self.diacritized + test_data.undiacritized)
            for token in sentence
        ]
        
        # Create the vocabulary.
        counts = Counter(words)
        
        vocab = {'w2i': None, 'i2w': None}
        
        # Sort the vocabulary.
        vocab['w2i'] = {
            word: index + 4 for index, (word, _) in enumerate(
                sorted(counts.items(), key=lambda x: x[1], reverse=True)
            )
        }
        
        # Add the keywords to the beginning.
        keywords = {'<sos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3}
        vocab['w2i'].update(keywords)
        
        # Create the inverse vocabulary.
        vocab['i2w'] = {index: word for word, index in vocab['w2i'].items()}
        
        # Update the vocabularies.
        self.vocab = vocab
        test_data.vocab = vocab
        
        return vocab
        
    def __len__(self):
        return len(self.undiacritized)
    
    def __getitem__(self, index):
        if not self.vocab:
            return(
                self.undiacritized[index],
                self.diacritized[index] if self.diacritized else None
            )
        
        if index >= len(self):
            print('\033[91mIndex out of range\033[0m')
            return None, None
        
        return(
            torch.tensor(self.undiacritized[index], dtype=torch.long),
            torch.tensor(self.diacritized[index], dtype=torch.long) if self.diacritized else None
        )