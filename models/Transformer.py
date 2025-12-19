import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F
import math
import copy

from dataclasses import dataclass
import binary_layers


@dataclass
class TransformerConfig:
    n_embd: int      = 768  # Embedding dimension
    n_head: int      = 12   # Number of attention heads
    n_layer: int     = 12  # Number of layers in the Transformer
    vocab_size: int  = 50304  # Size of the vocabulary
    dropout: float   = 0.0  # Dropout rate
    max_len: int     = 1024  # Maximum sequence length
    bias: bool       = False  # Whether to use bias in linear layers
    
    mlp_proj: type   = nn.Linear
    qkv_proj: type   = nn.Linear 
    c_proj: type =   nn.Linear

    n_binary: int = 0
    
    def __post_init__(self):
        # Ensures that `n_head` divides `n_embd`
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
            
def sdpa_mixed_dtype(q, k, v, **kw):
    # run the fast kernel in half precision
    q16, k16, v16 = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
    out = F.scaled_dot_product_attention(q16, k16, v16, **kw)
    return out.to(q.dtype)
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.qkv_proj = config.qkv_proj(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj   = config.c_proj(self.n_embd, self.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)
 
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.qkv_proj(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = sdpa_mixed_dtype(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = config.mlp_proj(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = config.mlp_proj(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config):

        super().__init__()
        
        self.pos_embd = nn.Embedding(config.max_len, config.n_embd)
        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)


        self.blocks = nn.ModuleList()

        for layer_idx in range(config.n_layer):
            if layer_idx < config.n_binary:
                config = copy.copy(config)
                config.mlp_proj = binary_layers.Linear
                config.qkv_proj = binary_layers.Linear
                config.c_proj = binary_layers.Linear
                self.blocks.append(Block(config))
            else:
                self.blocks.append(Block(config))

        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.head.weight = self.tok_embd.weight
        
        self.config = config

        self.apply(self._init_weights) 
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.01/math.sqrt(2 * config.n_layer))
            
        self.update_cache()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)   

    def update_cache(self):
        self.apply(self._update_cache)
    def _update_cache(self, module):
        if isinstance(module, binary_layers.Linear):
            module.update_cache()

    def forward(self, idx):

        
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.tok_embd(idx) 
        pos_emb = self.pos_embd(pos)

        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.head(x)
        return x
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: torch.device | str | None = None,
    ) -> str:
        """
        Autoregressively generate up to `max_new_tokens` new tokens, ensuring input dimensions are divisible by 8.
    
        Args
        ----
        prompt          : The input prompt as a string.
        tokenizer       : Any tokenizer that implements `encode` and `decode`.
        max_new_tokens  : How many tokens to append at most.
        temperature     : 1.0 = no scaling; <1.0 makes the distribution sharper,
                          >1.0 makes it flatter. Set to 0 for greedy decoding.
        top_k           : Keep only the `top_k` most-probable logits each step
                          (nucleus / top-p can be added similarly).
        device          : Override device; defaults to model device.
    
        Returns
        -------
        A string containing the original prompt followed by the generated text.
        """
        if device is None:
            device = next(self.parameters()).device
    
        self.eval()  # switch to inference mode
    
        # Encode prompt -> tensor shape [1, T]
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, :]  # (1, T)
    
        for _ in range(max_new_tokens):
            # 1) If sequence is longer than max_len, keep only the last max_len tokens
            if idx.size(1) > self.config.max_len:
                idx = idx[:, -self.config.max_len:]
    
            # Pad idx so its length is divisible by 8
            padding_length = (8 - idx.size(1) % 8) % 8  # Calculate necessary padding
            if padding_length > 0:
                padding = torch.zeros((idx.size(0), padding_length), dtype=torch.long, device=device)
                padded_idx = torch.cat([idx, padding], dim=1)  # Add padding
            else:
                padded_idx = idx
    
            # 2) Forward pass → logits for all positions
            logits = self(padded_idx)                   # (1, T_padded, vocab)
            logits = logits[:, idx.size(1) - 1, :]      # (1, vocab) only last "true" token
    
            # 3) Sample / pick next token
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature
                if top_k is not None:
                    # zero-out everything but the top-k logits
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)                    # (1, vocab)
                next_token = torch.multinomial(probs, num_samples=1)     # (1, 1)
    
            # 4) Append and continue
            idx = torch.cat([idx, next_token], dim=1)                    # (1, T+1)
    
        # Decode back to text
        return tokenizer.decode(idx[0].tolist())