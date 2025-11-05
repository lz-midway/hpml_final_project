import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLinear(nn.Module):
    """
    Binary-Normalised Fully-Connected Layer (BNFCL).

    All parameters are stored in fp32 but binarised (0/1) on the forward path.
    After the affine transform, features are normalised to zero-mean/unit-std
    per example, then an optional activation is applied.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 activation: str | None = None,
                 eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()
        self.activation = self._get_act(activation)
        self.eps = eps

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ---------- helpers --------------------------------------------------- #
    @staticmethod
    def _get_act(name):
        if name is None or name == "linear":
            return lambda x: x
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        raise ValueError(f"Unknown activation {name}")

    @staticmethod
    def _binarise(w: torch.Tensor):
        """Binary quantisation: 1 if w > mean(w) else 0 (per tensor)."""
        thresh = w.mean()
        return (w > thresh).float()

    # ---------- forward ---------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, *, in_features)
        returns: shape (batch, *, out_features)
        """
        w_q = self._binarise(self.weight)
        # Straight-Through Estimator for gradients
        if self.training:
            w_eff = self.weight + (w_q - self.weight).detach()
            b_eff = None
            if self.bias is not None:
                b_q = self._binarise(self.bias)
                b_eff = self.bias + (b_q - self.bias).detach()
        else:  # inference: use discrete params
            w_eff = w_q
            b_eff = self._binarise(self.bias) if self.bias is not None else None

        # Affine transform
        z = F.linear(x, w_eff, b_eff)        # ... shape (batch, *, out_features)

        # Per-example normalisation  (last dimension)
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True, unbiased=False)
        z = (z - mean) / (std + self.eps)

        # Activation
        return self.activation(z)


class BEMB(nn.Module):
    """
    Binary Normalised Embedding Layer (Algorithm 4).
    Produces token + positional embeddings with binary parameters.
    """

    def __init__(self,
                 max_len: int,
                 vocab_size: int,
                 emb_dim: int):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # Binary-normalised fully-connected layers for token & position lookup
        self.token_fc = BinaryLinear(vocab_size, emb_dim, bias=True, activation="linear")
        self.pos_fc   = BinaryLinear(max_len,    emb_dim, bias=True, activation="linear")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : LongTensor of shape (batch, seq_len) with token indices
        returns: Tensor of shape (batch, seq_len, emb_dim)
        """
        B, L = seq.shape
        device = seq.device

        # ---- Position embeddings ---- #
        pos_idx = torch.arange(L, device=device)                   # (L,)
        pos_onehot = F.one_hot(pos_idx, num_classes=self.max_len).float()  # (L, max_len)
        pos_emb = self.pos_fc(pos_onehot)                          # (L, emb_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)           # (B, L, emb_dim)

        # ---- Token embeddings ---- #
        tk_onehot = F.one_hot(seq, num_classes=self.vocab_size).float()    # (B, L, vocab_size)
        tk_emb = self.token_fc(tk_onehot)                          # (B, L, emb_dim)

        # ---- Combine ---- #
        tk_pos_emb = tk_emb + pos_emb                              # (B, L, emb_dim)
        return tk_pos_emb
