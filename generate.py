import torch
import contextlib
import json
from src.toy_qwen3MoE.model import Qwen3MoE

#select device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#load character vocabulary
with open("char_vocab.json", "r") as f:
  chars = json.load(f)

#create character-index mapping
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

#converts tensor of indicies into string
def decode(tensor):
  return ''.join([idx_to_char[i.item()] for i in tensor])

cfg = {
  "vocab_size": len(chars),   #number of unique characters(tokens)
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
  "qk_norm": False,            #will be computed as emb_dim // n_heads
  "dtype": torch.float32 if device == "mps" else torch.bfloat16,         #precision used in model (bfloat16 or float32)
}

cfg["head_dim"] = cfg["emb_dim"] // cfg["n_heads"]

def generate(model, prompt, max_new_tokens=100, temperature=0.85):
  model.eval() #put model in inference mode

  #convert input prompt string to tensor of token indices
  encoded = torch.tensor(
    [char_to_idx.get(c, 0) for c in prompt], 
    dtype=torch.long
  ).unsqueeze(0).to(device)

  autocast_context = (
    torch.autocast(device_type="cuda", dtype=cfg["dtype"])
    if torch.cuda.is_available() else contextlib.nullcontext()
  )

  for _ in range(max_new_tokens):
    input_ids = encoded[:, -cfg["context_length"]:]
    with torch.no_grad(), autocast_context:
      logits = model(input_ids)
      logits = logits[:, -1, :] / temperature
      probs = torch.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
    encoded = torch.cat([encoded, idx_next], dim=1)

  return decode(encoded[0])

model = Qwen3MoE(cfg).to_empty(device=device)
#PLEASE MAKE SURE YOU LOAD THE RIGHT MODEL ACCORIDNG TO THE EPOCHS YOU USED AND THE NAME WITH WHICH MODEL IS SAVED
model.load_state_dict(torch.load("toy_qwen3MoE_24epochs.pth"))
model.eval()

print("✅ Model loaded and ready for generation!\n")

prompt = "To be or not to be, that is the"
print(f"Prompt: {repr(prompt)}")
print("Generated:", repr(generate(model, prompt)))