import torch
from torch import nn
from torch.nn import functional as F
import mmap
import random
from torchinfo import summary
import matplotlib.pyplot as plt

import os

"""
This model will serve as a pre-train model for something greater later.

A GPT model. 
Will be pre-trained on ~40gb text from reddit.

Attention mechanism: Multi head self attention.
"""


# True -- IF YOU WANT TO TRAIN THE MODEL.
# False -- IF YOU WANT TO GENERATE WITH MODEL.
TRAIN_MODEL = False

device = "cuda" if torch.cuda.is_available() else "cpu"

BOOK_DATA_PATH = "V:\\PyTorch_LLM\\gpt_study\\data\\"
TRAIN_DATA_PATH = "V:\\PyTorch_LLM\\gpt_study\\data\\output_train.txt"
TEST_DATA_PATH = "V:\\PyTorch_LLM\\gpt_study\\data\\output_test.txt"
VOCAB_DATA_PATH = "V:\\PyTorch_LLM\\gpt_study\\data\\vocab.txt"
SAVE_PATH = "V:\\PyTorch_LLM\\gpt_study\\gpt_save_f\\"

# Model params
N_EMBD = 384 # How long is the vector per head.
N_LAYER = 16 # How many encode layers.
N_HEAD = 16 # How many heads in single multihead.
HEAD_SIZE = N_EMBD // N_HEAD # one head size.
BLOCK_SIZE = 128 # How long is one training sample.
BATCH_SIZE = 64 # How many batches per sample.

# baseline model (53.1M trainable params, video ram: ~11gb, gpu heat: ~70c):
# model params. embd> 384, layer> 16, head> 16, blocksize> 128, batch>64
# train params. eval> 16, dropout> 0.2, LR>3e-4

# openai gpt 1: / n_embd 768, layers 12, heads 12
# gophers: / n_embd 512, layers 8, heads 16
# fyi: openai gpt 3 has 96 heads.

# Train params
EVAL = 16
DROPOUT = 0.2
EPOCH = 1000
LR = 3e-4

# used for loss tracking and plotting.
train_loss_list = []
train_epoch_list = []
test_loss_list = []
test_epoch_list = []

# Init vocab from the vocab datafile.
chars = ""
with open(VOCAB_DATA_PATH, 'r', encoding="utf-8") as f:
    text = f.read()
    chars = sorted(list(set(text)))
    f.close()
vocab_size = len(chars)

def encode(s):
    string_to_int = { ch:i for i, ch in enumerate(chars) }
    return [string_to_int.get(c, 0) for c in s]

def decode(l):
    int_to_string = { i:ch for i, ch in enumerate(chars) }
    return "".join([int_to_string.get(i, '') for i in l])

def getData(pth : str):
    """
    pth: Path to datafile.\n    
    Select random chunk from datafile and read it. BLOCK_SIZE*BATCH_SIZE-1 = size of the chunk. \n
    Turns random chunk into a torch.tensor and returns X, and y for each BATCH
    """
    with open(pth, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            size = len(mm)
            start_pos = random.randint(0, (size) - BLOCK_SIZE*BATCH_SIZE)
            mm.seek(start_pos)
            block = mm.read(BLOCK_SIZE*BATCH_SIZE-1)
            block = block.decode("utf-8", errors="ignore").replace("\r", "")
            data = torch.tensor(encode(block), dtype=torch.long)
        mm.close()
    f.close()
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class multihead_attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        w = torch.matmul(key, query.transpose(-2, -1))
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        w = torch.matmul(w, value)

        return w


class block(nn.Module):    
    def __init__(self):
        super().__init__()

        self.attention = nn.ModuleList([multihead_attention() for _ in range(N_HEAD)])
        self.norm1 = nn.LayerNorm(N_EMBD)
        self.proj = nn.Linear(HEAD_SIZE * N_HEAD, N_EMBD)
        self.norm2 = nn.LayerNorm(N_EMBD)
        self.feedForward = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD)
        )
        
        self.norm3 = nn.LayerNorm(N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        y = self.norm1(x)
        y = torch.cat([head(y) for head in self.attention], dim=-1)
        y = self.proj(y)
        y = self.dropout(y)

        x = x + y
        x = self.norm2(x)

        y = self.feedForward(x)
        y = self.dropout(y)

        x = x + y

        return x


class GPT_model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

        self.dropout = nn.Dropout(DROPOUT)

        self.tranformer_block = nn.Sequential(*[block() for _ in range(N_LAYER)])

        self.norm1 = nn.LayerNorm(N_EMBD)
        self.linear1 = nn.Linear(N_EMBD, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        emb_token = self.token_embedding_table(x)
        emb_pos = self.position_embedding_table(torch.arange(T, device=device))

        x = emb_token + emb_pos
        x = self.tranformer_block(x)
        x = self.norm1(x)
        x = self.linear1(x)

        return x

def train(model, loss_fn, optimizer):
    model.train()
    x, y = getData(TRAIN_DATA_PATH)
    logits = model(x)

    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    y = y.view(B*T)

    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def test(model, loss_fn):
    model.eval()
    with torch.inference_mode():
        loss_count = 0
        for _ in range(EVAL):
            x, y = getData(TEST_DATA_PATH)
            logits = model(x)

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)

            loss = loss_fn(logits, y)
            loss_count += loss.item()

        loss_count /= EVAL
        return loss_count
    
def generate(model, context : torch.Tensor, max_tokens : int):
    context = context.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(max_tokens):

            logits = model(context)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_index = torch.multinomial(probs, num_samples=1)

            context = torch.cat((context, next_index), dim=1)

            if context.shape[1] >= BLOCK_SIZE+1:
                context = torch.cat((context[:,:0], context[:,1:]), dim=1)

    context = context[0]
    context = context.tolist()
    result = decode(context)
    print("Tokens generated: ", len(result))

    return result

def main():
    model = GPT_model(vocab_size)

    if os.path.exists(SAVE_PATH + 'TOP_V3_GPT_MODEL_STATE_DICT.pth'):
        model.load_state_dict(torch.load(SAVE_PATH + 'TOP_V3_GPT_MODEL_STATE_DICT.pth'))
        print("[INFO] - LOADED STATE DICT FROM PATH: " + SAVE_PATH + 'TOP_V3_GPT_MODEL_STATE_DICT.pth')
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_test_loss = 9999999999
    for epoch in range(EPOCH):
        train_loss = train(model, loss_fn, optim)
  #      collectStatistic(epoch, train_loss.item(), lossType="train")

        if epoch % 100 == 0 or epoch+1 == EPOCH:
            print(f"Epoch: {epoch} | training loss: {train_loss.item()}")
            collectStatistic(epoch, train_loss.item(), lossType="train")

            if epoch > int(EPOCH/1.5):
                test_loss = test(model, loss_fn)        
                collectStatistic(epoch, test_loss, lossType="test")
                print(f"Epoch: {epoch} | testing loss: {test_loss}")

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_epoch = epoch

                    torch.save(model, SAVE_PATH + 'TOP_V3_GPT_OWN_MODEL_MODEL.pth')
                    torch.save(model.state_dict(), SAVE_PATH + 'TOP_V3_GPT_MODEL_STATE_DICT.pth')
                 
    print(f'Best model found at epoch: {best_test_epoch}. Model saved in path {SAVE_PATH}')



def soloGenerate(text, max_tokens):
    model = GPT_model(vocab_size)
    model = model.to(device)
    model.load_state_dict(torch.load(SAVE_PATH + 'TOP_V3_GPT_MODEL_STATE_DICT.pth'))
    print("[INFO] - LOADED STATE DICT FROM PATH: " + SAVE_PATH + 'TOP_V3_GPT_MODEL_STATE_DICT.pth')
    model.to(device)

    context = torch.tensor(encode(text), dtype=torch.long)
    context = context.unsqueeze(0)
    context = context.to(device)

    out = generate(model, context, max_tokens)


#    train_sample, _ = getData(TRAIN_DATA_PATH)
#    summary(model=model, input_data=train_sample,
#        col_names=["input_size", "output_size", "num_params", "trainable"])


    return out

def collectStatistic(epoch, loss, lossType):
    if lossType == "train":
        train_loss_list.append(loss)
        train_epoch_list.append(epoch)
    else:
        test_loss_list.append(loss)
        test_epoch_list.append(epoch)


def plotLosses(epoch_list, loss_list, color):
    if color == "red":
        plt.plot(epoch_list, loss_list, color="red", linestyle="-")

    elif color == "blue":
        plt.plot(epoch_list, loss_list, color="blue", linestyle="-")


if __name__ == "__main__":
    if TRAIN_MODEL:
        main()
        plotLosses(train_epoch_list, train_loss_list, color="red")
        plotLosses(test_epoch_list, test_loss_list, color="blue")
        plt.show()

        

    else:
        print(soloGenerate("My name is Vex and I'm ", max_tokens=100), "\n")













