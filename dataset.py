# Author: Bora Boyacıoğlu
# Student ID: 150200310

import csv
import pickle as pkl
from unidecode import unidecode

from torch.utils.data import Dataset

class DiacritizationDataset(Dataset):
    def __init__(self, file_name, type):
        # Read the data from the file.
        text: list = self.read_data(file_name)
        
        # Initialize the diacritized and undiacritized data.
        self.diacritized = None
        self.undiacritized = None
        
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
        
    def read_data(self, file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                # Read the CSV file.
                reader = csv.reader(f)
                text = []
                for row in reader:
                    text.append(row[1])
                return text[1:]
        except FileNotFoundError:
            print(f'\033[91mFile {file_name} not found\033[0m')
            return []
    
    def undiacritize(self, text):
        return [unidecode(sentence) for sentence in text]
    
    def get(self, idx, type):
        if type == 'd':
            return self.diacritized[idx] if self.diacritized else None
        elif type == 'und':
            return self.undiacritized[idx]
        else:
            print('\033[91mInvalid type\033[0m')
            return None
    
    def set(self, idx, type, new):
        if type == 'd':
            self.diacritized[idx] = new
        elif type == 'und':
            self.undiacritized[idx] = new
        else:
            print('\033[91mInvalid type\033[0m')
    
    def save_data(self, file_name):
        with open(file_name, 'wb') as f:
            pkl.dump(self, f)
        
    def __len__(self):
        return len(self.undiacritized)
    
    def __getitem__(self, idx):
        if idx >= len(self.undiacritized):
            print('\033[91mIndex out of range\033[0m')
            return None
        return {
            'diacritized': self.diacritized[idx] if self.diacritized else None,
            'undiacritized': self.undiacritized[idx]
        }