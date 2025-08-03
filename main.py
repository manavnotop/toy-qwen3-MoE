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

def apply_rope(x, cos, sin):
  # x: (batch_size, num_heads, seq_len, head_dim)
  batch_size, num_heads, seq_len, head_dim = x.shape
  assert head_dim % 2 == 0, "Embedding must be even" 

  x1 = x[..., : head_dim // 2]
  x2 = x [..., head_dim //2 :]

  cos = cos[:seq_len, :].unsqeeze(0).unsqueeze(0)
  sin = sin[:seq_len, :].unsqeeze(0).unsqueeze(0)

  rotated = torch.cat([-x2, x1], dim=-1)
  x_rotated = x * cos + rotated * sin

  return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
  def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
    super().__init__()
    assert num_heads % num_kv_groups == 0, "num heads must be divisible by num kv groups"

    self.num_heads = num_heads
    self.num_kv_groups = num_kv_groups
    self.group_size = num_heads // num_kv_groups

    if head_dim is None:
      assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
      head_dim = d_in // num_heads

    self.head_dim = head_dim
    self.d_out = num_heads * head_dim

    self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
    self.W_key = nn. Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
    self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

    self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    if qk_norm:
      self.q_norm = RMSNorm(head_dim, eps=1e-6)
      self.k_norm = RMSNorm(head_dim, eps=1e-6)
    else:
      self.q_norm = self.k_norm = None

  def forward(self, x, mask, cos, sin):
    b, num_tokens = x.shape

    queries = self.W_query(x)
    keys = self.W_key(x)
    values = self.W_value(x)

    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
    values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

    if self.q_norm:
      queries = self.q_norm(queries)
    if self.k_norm:
      keys = self.k_norm(keys)

    #apply rope
    queries = apply_rope(queries, cos, sin)
    keys = apply_rope(keys, cos, sin)

    # Expand K and V to match number of heads
    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    # Attention
    attn_scores = queries @ keys.transpose(2, 3)
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

    context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
    return self.out_proj(context)
  
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = GroupedQueryAttention(
      d_in=cfg["emb_dim"],
      num_heads=cfg["n_heads"],
      head_dim=cfg["head_dim"],
      num_kv_groups=cfg["n_kv_groups"],
      qk_norm=cfg["qk_norm"],
      dtype=cfg["dtype"]
    )
    if cfg["num_experts"] > 0:
      self.ff = MoEFeedForward(cfg)
    else:
      self.ff = FeedForward(cfg)
    self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
    self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

  def forward(self, x, mask, cos, sin):
    # Shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
    x = x + shortcut  # Add the original input back

    # Shortcut connection for feed-forward block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = x + shortcut  # Add the original input back

    return x
  
class Qwen3MoE(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    # Main model parameters
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

    self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
      [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )

    self.final_norm = RMSNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    # Reusuable utilities
    if cfg["head_dim"] is None:
      head_dim = cfg["emb_dim"] // cfg["n_heads"]
    else:
      head_dim = cfg["head_dim"]
    cos, sin = compute_rope_params(
      head_dim=head_dim,
      theta_base=cfg["rope_base"],
      context_length=cfg["context_length"]
    )
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
    self.cfg = cfg


  def forward(self, in_idx):
    # Forward pass
    tok_embeds = self.tok_emb(in_idx)
    x = tok_embeds

    num_tokens = x.shape[1]
    mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
    
    for block in self.trf_blocks:
      x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x.to(self.cfg["dtype"]))
    return logits