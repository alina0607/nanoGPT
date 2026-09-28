from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# ------------

torch.manual_seed(1995)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# 相對於「這支 .py 檔」,不是相對於終端機所在的資料夾
DATA_PATH = Path(__file__).parent / 'data' / 'tinyshakespeare' / 'input.txt'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
# f.close()


# here are all the unique characters that occur in this text
chars = sorted(set(text))
vocab_size = len(chars)

# create a mapping from characters to integers

char_to_id = { ch:i for i, ch in enumerate(chars) }
id_to_char = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [char_to_id[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([id_to_char[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    #random batch_size counts of start position
    #tuple: (batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size,))


    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# super simple bigram model
#inherit
class BigramLanguageModel(nn.Module):


    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        print('token_embedding_table : ' , self.token_embedding_table)
    


    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


