import torch
import torch.nn as nn
from torch.nn import functional as F


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
with open('data/tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
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

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
