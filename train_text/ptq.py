"""
Full-precision vs PTQ (INT8/INT4) on GPU using bitsandbytes
Usage: python ptq.py --ckpt path/to/full_model.pt
"""
import os, sys, time, argparse, torch, torch.nn as nn
from torch.nn import LayerNorm, functional as F
from dataclasses import dataclass
from data import text_dataset
import bitsandbytes as bnb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, 
                    help="Path to full-precision checkpoint (.pt file or directory containing it)")
args = parser.parse_args()

device, eval_iters = 'cuda', 200
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)
_, val_loader = text_dataset.get_loader(20, 1024, 4, 8)

# ---------- Model ----------
@dataclass
class TransformerConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    vocab_size: int = 50304
    dropout: float = 0.0
    max_len: int = 1024
    bias: bool = False

def sdpa_mixed_dtype(q, k, v, **kw):
    q16, k16, v16 = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
    return F.scaled_dot_product_attention(q16, k16, v16, **kw).to(q.dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_embd, self.dropout = config.n_head, config.n_embd, config.dropout
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = sdpa_mixed_dtype(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        return self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x): return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

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
        self.config = config
        self.pos_embd = nn.Embedding(config.max_len, config.n_embd)
        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.head.weight = self.tok_embd.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx):
        x = self.tok_embd(idx) + self.pos_embd(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(x)

def load_fp(ckpt):
    c = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = TransformerConfig()
    m = Transformer(cfg)
    sd = {k.replace('module._orig_mod.', '').replace('_orig_mod.', ''): v for k, v in c['model'].items()}
    m.load_state_dict(sd)
    return m.to(device)

def quantize_bnb_int8(model):
    """Replace Linear with bitsandbytes Int8 Linear (GPU)"""
    import copy
    model = copy.deepcopy(model).cpu()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'head' not in name and 'embd' not in name:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            old = getattr(parent, parts[-1])
            # Create Int8 linear
            new_linear = bnb.nn.Linear8bitLt(
                old.in_features, old.out_features,
                bias=old.bias is not None,
                has_fp16_weights=False,
                threshold=6.0
            )
            new_linear.weight = bnb.nn.Int8Params(old.weight.data, requires_grad=False)
            if old.bias is not None:
                new_linear.bias = nn.Parameter(old.bias.data)
            setattr(parent, parts[-1], new_linear)
    return model.to(device)

def quantize_bnb_int4(model):
    """Replace Linear with bitsandbytes NF4 Linear (GPU)"""
    import copy
    from bitsandbytes.nn import Linear4bit, Params4bit
    model = copy.deepcopy(model).cpu()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'head' not in name and 'embd' not in name:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            old = getattr(parent, parts[-1])
            # Create 4bit linear (NF4 quantization)
            new_linear = Linear4bit(
                old.in_features, old.out_features,
                bias=old.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type='nf4'
            )
            new_linear.weight = Params4bit(old.weight.data, requires_grad=False, quant_type='nf4')
            if old.bias is not None:
                new_linear.bias = nn.Parameter(old.bias.data)
            setattr(parent, parts[-1], new_linear)
    return model.to(device)

def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def count_quantized_layers(model):
    int8 = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear8bitLt))
    int4 = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    regular = sum(1 for m in model.modules() if type(m) == nn.Linear)
    return {'int8': int8, 'int4': int4, 'regular': regular}

def loss_fn(l, t): return F.cross_entropy(l.view(-1, l.size(-1)), t.view(-1), ignore_index=-100)

@torch.no_grad()
def evaluate(m, name):
    m.eval()
    losses, times = [], []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters: break
        x, y = x.to(device), y.to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        with ctx: logits = m(x)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(loss_fn(logits, y).item())
    loss, ppl = sum(losses)/len(losses), torch.exp(torch.tensor(sum(losses)/len(losses))).item()
    ms, mb = sum(times)/len(times)*1000, get_model_size_mb(m)
    q = count_quantized_layers(m)
    q_str = f" ({q['int8']} int8)" if q['int8'] else (f" ({q['int4']} nf4)" if q['int4'] else "")
    print(f"{name:<25} Loss={loss:.4f} PPL={ppl:.2f} {ms:.1f}ms {mb:.1f}MB{q_str}")
    return {"name": name, "loss": loss, "ppl": ppl, "ms": ms, "mb": mb}

def main():
    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        pts = [f for f in os.listdir(ckpt) if f.endswith(".pt")]
        if len(pts) == 0:
            print(f"No .pt file found in directory: {ckpt}")
            return
        ckpt = os.path.join(ckpt, pts[0])
    if not os.path.exists(ckpt): print(f"Missing: {ckpt}"); return
    results = []
    print("\n" + "="*70 + "\n GPU QUANTIZATION COMPARISON (bitsandbytes)\n" + "="*70)

    print("\n[1/3] Full precision...")
    results.append(evaluate(load_fp(ckpt), "FP32/BF16"))
    torch.cuda.empty_cache()

    print("\n[2/3] bitsandbytes INT8...")
    m = load_fp(ckpt)
    q8 = quantize_bnb_int8(m)
    results.append(evaluate(q8, "INT8"))
    del q8; torch.cuda.empty_cache()

    print("\n[3/3] bitsandbytes NF4...")
    m = load_fp(ckpt)
    q4 = quantize_bnb_int4(m)
    results.append(evaluate(q4, "NF4"))
    del q4; torch.cuda.empty_cache()

    print("\n" + "="*70 + "\n SUMMARY (relative to FP32)\n" + "-"*70)
    base = results[0]
    for r in results[1:]:
        dp = (r['ppl'] - base['ppl']) / base['ppl'] * 100
        ds = (r['mb'] - base['mb']) / base['mb'] * 100
        print(f"  {r['name']}: PPL {dp:+.1f}%, Size {ds:+.1f}%")
    print("="*70)

if __name__ == "__main__": main()
