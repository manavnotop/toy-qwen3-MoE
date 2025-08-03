import torch
import torch.nn as nn

class RMSNorm(nn.Module):
  def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
    super().__init__()
    self.eps = eps
    self.qwen3_compatible = qwen3_compatible
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zero(emb_dim)) if bias else None

  def forward(self, x):
    input_type = x.dtype

    if self.qwen3_compatible:
      x = x.to(torch.float32)
    
    #next 3 lines are main maths, rest is optimisation for better calcultion
    mean_square = x.pow(2).mean(dim=-1, keep_dim=True)
    norm_x = x * torch.rsqrt(mean_square + self.eps)
    norm_x = norm_x * self.scale

    if self.shift is not None:
      norm_x = norm_x + self.shift

    return norm_x.to(input_type)
  
def compute_rope_params(head_dim, theta_base=10000, context_length=4096, dtype=torch.float32):
  assert head_dim % 2 == 0, "Embeddings must be even"

  even_indices = torch.arange(0, head_dim, 2, dtype=dtype)
  powers = even_indices[: head_dim // 2]
  normalised_powers = powers / head_dim 

  inv_freq = 1 / (theta_base ** powers)

  position = torch.arange(context_length, dtype=dtype)

  freqeuncies = inv_freq.unsqueeze(0)
  position = position.unsqueeze(1)

  angles = position * freqeuncies
  angles = torch.cat([angles, angles], dim=1)

  cos = torch.cos(angles)
  sin = torch.sin(angles)

  return cos, sin

def apply_rope(x, cos, sin, offset=0):
  # x: (batch_size, num_heads, seq_len, head_dim)
  batch_size, num_heads, seq_len, head_dim = x.shape
  assert head_dim % 2 == 0, "Embedding must be even" 

  x1 = x[..., : head_dim // 2]
  x2 = x [..., head_dim //2 :]

  cos = cos[offset:offset + seq_len, :].unsqeeze(0).unsqueeze(0)
  sin = sin[offset:offset + seq_len, :].unsqeeze(0).unsqueeze(0)

  rotated = torch.cat([-x2, x1], dim=-1)
  x_rotated = x * cos + rotated * sin

  return x_rotated.to(dtype=x.dtype)