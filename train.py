import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import contextlib

from src.toy_qwen3MoE.model import Qwen3MoE

#load the dataset
print("Loading dataset...")
with open("data/tiny_shakespeare.txt", "r") as f:
  text = f.read()

#build character level vocabulary
chars = sorted(list(set(text)))
print("chars =", repr(chars))
vocab_size = len(chars)

#mapping characters and token indicies
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(s):
  return torch.tensor([char_to_idx[c] for c in s if c in char_to_idx], dtype=torch.long)

def decode(tensor):
  return ''.join([idx_to_char[i.item()] for i in tensor])

#encode dataset into tokens ids
data = encode(text)
print(f"Dataset loaded: {len(text):,} chars, vocab_size={vocab_size}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#uses float32 for MPS(better on MPS), otherwise bfloat16
cfg_dtype = torch.float32 if device == "mps" else torch.bfloat16

cfg = {
  "vocab_size": vocab_size,        # Number of unique tokens (characters)
  "emb_dim": 128,                  # Token embedding dimension
  "n_heads": 4,                    # Number of attention heads
  "n_kv_groups": 2,                # Number of key/value head groups
  "n_layers": 4,                   # Number of Transformer blocks
  "hidden_dim": 128,              # FFN hidden size (used if not using MoE)
  "moe_intermediate_size": 128,   # MoE expert hidden dimension
  "num_experts": 2,               # Total experts in MoE
  "num_experts_per_tok": 1,       # Top-k experts per token
  "context_length": 128,          # Maximum input length
  "rope_base": 10000.0,           # Rotary embedding base
  "qk_norm": False,               # Use RMSNorm on Q/K (optional)
  "dtype": cfg_dtype              # Data type (float32 for MPS, bfloat16 elsewhere)
}

cfg["head_dim"] = cfg["emb_dim"] // cfg["n_heads"]

model = Qwen3MoE(cfg).to_empty(device=device)

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
  elif isinstance(m, nn.Embedding):
    nn.init.normal_(m.weight, mean=0.0, std=0.02)

model.apply(init_weights)

if device != "mps":
  model = model.to(cfg["dtype"])

optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

seq_len = cfg["context_length"]
batch_size = 4
epochs = 24
steps_per_epoch = 250
losses = []

model.train()
print(f"\nStarting training: {epochs} epochs × {steps_per_epoch} steps")

#use automatic mixed precision if using CUDA
use_autocast = torch.cuda.is_available()
autocast_context = (
  torch.autocast(device_type="cuda", dtype=cfg["dtype"])
  if use_autocast else contextlib.nullcontext()
)

#training loop
for epoch in range(epochs):
  start_time = time.time()
  epoch_loss = 0.0
  progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")

  for step in progress_bar:
    # Sample random sequences
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    xb = torch.stack([data[i:i+seq_len] for i in ix]).to(device).long()
    yb = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device).long()

    optimizer.zero_grad()

    #forward pass
    with autocast_context:
      logits = model(xb)
      loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    epoch_loss += loss.item()
    progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

  avg_loss = epoch_loss / steps_per_epoch
  losses.append(avg_loss)
  elapsed = time.time() - start_time
  print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.3f} | Time: {elapsed:.0f}s")

  #sample generation
  model.eval()
  input_seq = xb[0].unsqueeze(0)
  generated = input_seq[0].tolist()
  with torch.no_grad():
    for _ in range(100):  # generate 100 tokens
      input_chunk = torch.tensor(generated[-seq_len:], device=device).unsqueeze(0)
      logits = model(input_chunk)
      probs = torch.softmax(logits[:, -1, :], dim=-1)
      next_token = torch.multinomial(probs, num_samples=1).item()
      generated.append(next_token)
  print("📝 Sample generation:", repr(decode(torch.tensor(generated))))
  model.train()

print("\n🎉 Training complete! Model saved.")

#plot the loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss", marker="o")
plt.title("Toy Qwen3MoE - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
print("📉 Loss curve saved as 'loss_curve.png'")

model_save_path = f"toy_Qwen3MoE_{epochs}epochs.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'")

plot_save_path = f"assets/loss_curve_{epochs}epochs.png"
plt.savefig(plot_save_path)
print(f"Loss curve saved as '{plot_save_path}'")

import json
with open("char_vocab.json", "w") as f:
  json.dump(chars, f)