import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # cuantas secuencias en paralelo puede procesar el model
block_size = 8 # cual es la longitud maxima de contexto para las predicciones?
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # permite correr en GPU si es posible
eval_iters = 200
# ------------

torch.manual_seed(1337) # reproducibilidad

# Descargar los datos
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# El vocabulario
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoder y decoder
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train y test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% training, 10% validation
train_data = data[:n]
val_data = data[n:]

# Data loaders
def get_batch(split):
    # Generar un small batch the datos de varios x, secuencias de tokens
    # e y, shifteado en una unidad, para predecir el próximo valor.
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # Si hay GPU, hay que mover los datos al device para que se use. 
    return x, y

# La función que sigue estima la loss sobre varios batches. Es útil porque la medida
# puede ser muy ruidosa. Calculando los averages se observa mejor la optimización.
# El decorador abajo le dice a Pytorch que no haremos backprop en la función que sigue.
# De esta forma, no guarda los gradientes y la evaluación se hace mucho más rápida.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Le decimos al modelo que entre a inference
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Le decimos al modelo que vuelta a training
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

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
        # idx es (B, T) array de indices en el contexto actual
        for _ in range(max_new_tokens):
            # las predicciones
            logits, loss = self(idx)
            # nos concentramos solo en el ultimo time, predecir el next token
            logits = logits[:, -1, :] # se convierte en (B, C)
            # aplicamos softmax para obtener probabilidades
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sampleamos de la distribución
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # concatenamos el valor 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device) # Si hay GPU, movemos el modelo a device para que los weights the nn.Embedding tambien se procesen rápido

# Crear el optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # cada cierto tiempo evaluamos la pérdida en train y validation
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # muestrear un lote de datos
    xb, yb = get_batch('train')

    # evaluar la loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generar con el modelo 
context = torch.zeros((1, 1), dtype=torch.long, device=device) # El contexto también debe generarse en device GPU, si es que hay
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))