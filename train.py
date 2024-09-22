import torch
import torch.nn as nn
from model import HangmanFormer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import HangmanDataset
from loss import HangmanFormerLoss
from tqdm import tqdm

writer = SummaryWriter()
model = HangmanFormer(vocab_size=26)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** epoch)

# Read file and create dataset
with open('words_250000_train.txt', 'r') as f:
    words = f.read().splitlines()
dataset = HangmanDataset(words)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
epochs = 100
step = 0
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    loss_fn = HangmanFormerLoss()
    for batch in tqdm(dataloader):
        obscured_word, original_word, padding_mask = batch
        obscured_word, original_word, padding_mask = obscured_word.to(device), original_word.to(device), padding_mask.to(device)

        optimizer.zero_grad()
        queries = torch.nonzero(obscured_word == 26, as_tuple=False)
        output = model(obscured_word, padding_mask) 
        loss = loss_fn(output, original_word, queries)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss, step)

        total_loss += loss.item()
        step += 1

    scheduler.step()
    writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
    avg_loss = total_loss / len(dataloader)
    
    # Save .ckpt file
    if epoch % 10 == 0 and epoch > 0:
        print (f'Saving model to ckpts/hangman_model_{epoch}.pt')
        torch.save(model.state_dict(), f'ckpts/hangman_model_{epoch}.pt')
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

torch.save(model.state_dict(), 'ckpts/hangman_model_final.pt')