import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. MODEL CONFIG 1.24B PARAMETERS
# ==============================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 50280 # same as Tokenizer EleutherAI/gpt-neox-20b
    max_seq_len: int = 8192 #8K
    d_model: int = 2048 #2K
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        self.head_dim = self.d_model //self.n_heads
        assert self.d_model % self.n_heads == 0, "Model dimension must equally split amongs heads"
        assert self.n_heads % self.n_kv_heads == 0, "number of heads share KV must be integer"


# ==============================================================================
# 2. RMSNORM (Root Mean Square Layer Normalization)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x*torch.rsqrt(variance + self.eps) * self.weight


# ==============================================================================
# 3. ROTARY POSITION EMBEDDING (RoPE)
# ==============================================================================
def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0/(theta ** (torch.arange(0, head_dim, 2)[: (head_dim //2)].float()/head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
        xq: torch.Tensor, 
        xk: torch.Tensor, 
        freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Slice freqs_cis to match sequence length
    seqlen = xq.shape[1]
    freqs_cis = freqs_cis[:seqlen]

    # Apply RoPE to query and key tensors
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==============================================================================
# 4. SWIGLU FEED-FORWARD NETWORK (FFN)
# ==============================================================================
class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(8 * (config.d_model)/3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.wgate = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.wup = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.wdown = nn.Linear(hidden_dim, config.d_model, bias=False)

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: wup(SiLU(wgate(x)) * wdown(x))
        return self.wup(F.silu(self.wgate(x))) * self.wdown(x)

# ==============================================================================
# 5. GROUPED-QUERY ATTENTION (GQA) WITH FLASH ATTENTION
# ==============================================================================
class GroupQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads
        self.head_dim = config.head_dim

        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        #1. Linear projection
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        #2. Reshape
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        kv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Transpose 
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transponse(1, 2)

        # use FlashAttention 2
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p = 0.0,
            is_causal=True
        )

        output = output.transpose(1, 2).contingous().view(bsz, seqlen, -1)
        return self.wo(output)

# ==============================================================================
# 6. TRANSFORMER BLOCK
# ==============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, config: Module.Config):
        super().__init__()
        self.attention = GroupQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-Normalization Architecture
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        output = h + self.feedforward(self.ffn_norm(h));
        return output

# ==============================================================================
# 7. MODEL TRANSFORMER 1.24B COMPLETE
# ==============================================================================
class Transformer1B(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying
        self.tok_embeddings.weight = self.output.weight

        #Precompute RoPE table
        freqs_cis= precompute_freqs_cis(
            head_dim = config.head_dim,
            seq_len = config.max_seq_len,
            theta = config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        #Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optiona[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freq_cis = self.freqs_cis[:seqlen]

        # Forward through 24 Transformer Blocks
        for layer in self.layers:
            h = layer(h, self.freqs_cis)

        h = self.norm(h)

        if targets is not None:
            # if pass targets -> calculate Cross Entropy Loss
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # Inference mode: Calculate logits for last token
            logits = self.output(h[:, -1:, :])
            return logits, None

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

# ==============================================================================
# 8. SCRIPT (SELF-TEST)
# ==============================================================================
if __name__ == "__main__":
    print("start testing...")

    config = ModelConfig()
    model = Transformer1B(config)
    total_params = model.get_num_params()

    print(f"  - Vocab Size    : {config.vocab_size:,}")
    print(f"  - Hidden Dim    : {config.d_model}")
    print(f"  - Num Layers    : {config.n_layers}")
    print(f"  - Context Window: {config.max_seq_len:,}")
    print(f"  - Total Params: {total_params / 1e9:.2f} Billion ({total_params:,} parameters)")

    # Test Forward Pass B = 2 T = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n run Forward Pass on {device.upper()}...")

    model = model.to(device)
    dummy_x = torch.randint(0, config.vocab_size, (2, 512), device=device)
    dummy_y = torch.randint(0, config.vocab_size, (2, 512), device=device)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    with torch.autocast(device_type=device, dtype=dtype):
        logits, loss = model(dummy_x, dummy_y)

    print(f"  - Output Logits Shape: {logits.shape}")
    print(f"  - Sample Loss Value  : {loss.item():.4f}")
    print(f"  - Initial Loss: ln({config.vocab_size}) = {math.log(config.vocab_size):.2f}")
    print("\n🎉 All module work as expected!")
