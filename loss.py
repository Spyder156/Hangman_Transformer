import torch
import torch.nn as nn

class HangmanFormerLoss(nn.Module):
    def __init__(self):
        super(HangmanFormerLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, original_word, queries):
        output = output[queries[:, 0], queries[:, 1]] # B, V
        original_letters = original_word[queries[:, 0], queries[:, 1]] # B
        loss = self.loss_fn(output, original_letters.to(torch.long))
        return loss