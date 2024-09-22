import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import numpy as np

class HangmanDataset(Dataset):
    def __init__(self, words, context_size=60):
        self.context_size = context_size
        self.words = words
        self.data = self.create_data(words)

    def create_data(self, words):
        data = words
        return data

    def obscure_word(self, word):
        word_list = list(word)
        hidden_letters = []

        num_to_hide = random.randint(1, len(word))

        indices_to_hide = random.sample(range(len(word)), num_to_hide)

        for index in indices_to_hide:
            hidden_letters.append(word_list[index])
            word_list[index] = "_"

        obscured_word = ''.join(word_list)

        return obscured_word, hidden_letters

    def encode_context(self, word, guessed_letters):
        # Initialize the context tensor (1x60) with padding encoded as 27
        context_tensor = np.full((1, self.context_size), 27)

        # Encode the word with blanks
        for i, char in enumerate(word):
            if i >= 26:
                break  # Limit to 26 characters (a-z positions)
            if char in guessed_letters or char != "_":
                context_tensor[0, i] = ord(char) - ord('a')  # Encode as 0-25
            else:
                context_tensor[0, i] = 26  # Encode blank as 26

        # # Encode previous guesses
        # for i, char in enumerate(sorted(guessed_letters)):
        #     if i + 26 >= self.context_size:
        #         break  # Limit to 34 slots for guesses
        #     context_tensor[0, i + 26] = ord(char) - ord('a')  # Encode as 0-25

        return context_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.words[idx]
        obscured_word, hidden_letters = self.obscure_word(word)
        guessed_letters = set()
        obscured_word = torch.tensor(self.encode_context(obscured_word, guessed_letters), dtype=torch.int)
        original_word = torch.tensor(self.encode_context(word, guessed_letters), dtype=torch.int) 
        padding_mask = original_word == 27
        return obscured_word[0], original_word[0], padding_mask[0]