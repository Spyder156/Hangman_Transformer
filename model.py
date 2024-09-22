import torch
import torch.nn as nn

class HangmanFormer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=60):
        super(HangmanFormer, self).__init__()
        self.embedding = nn.Embedding(vocab_size+2, d_model)
        self.positional_encoding = self.create_positional_encoding(d_model, max_len)
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x, padding_mask):
        B, S = x.shape
        x = self.embedding(x) + self.positional_encoding[:, :S, :].to(x.device) # B, S, E
        x = self.transformer(x, src_key_padding_mask=padding_mask) # B, S, E
        x = self.fc(x) # B, S, V
        return x

if __name__ == '__main__':
    model = HangmanFormer(vocab_size=26)
    x = torch.randint(0, 26, (4, 60)) # B, C
    # model.eval()
    # with torch.no_grad():
    # nested_input = torch.nested.nested_tensor([torch.randn(10, 128), torch.randn(5, 128), torch.randn(7, 128), torch.randn(19, 128)])
    # print (model.transformer(nested_input))
    word_len = torch.randint(2, 10, (4,))
    padding_mask = torch.arange(10).unsqueeze(0).expand(4, 10) >= word_len.unsqueeze(1)
    # breakpoint()
    print(model(x, padding_mask).shape)
    