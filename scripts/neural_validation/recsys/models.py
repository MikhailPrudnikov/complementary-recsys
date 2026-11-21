"""Small shared content encoder, relation-specific towers, and pair Transformer."""

import torch
from torch import nn
from torch.nn import functional as F


class ContentEncoder(nn.Module):
    def __init__(self, tokens, categories, vocab_size, n_categories, dim=128):
        super().__init__()
        self.register_buffer("tokens", torch.as_tensor(tokens, dtype=torch.long))
        self.register_buffer(
            "categories", torch.as_tensor(categories, dtype=torch.long)
        )
        self.words = nn.Embedding(vocab_size, 96, padding_idx=0)
        self.category = nn.Embedding(n_categories, 32)
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, items):
        words = self.tokens[items]
        mask = words.ne(0).unsqueeze(-1)
        vectors = self.words(words)
        title = (vectors * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        return self.mlp(
            torch.cat([title, self.category(self.categories[items])], dim=-1)
        )


class TwoTower(nn.Module):
    def __init__(self, encoder, separate=False, dim=128):
        super().__init__()
        self.encoder = encoder
        self.separate = separate
        n = 2 if separate else 1
        self.queries = nn.ModuleList([nn.Linear(dim + 32, dim) for _ in range(n)])
        self.candidates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])
        self.log_scale = nn.Parameter(torch.full((n,), 2.302585))
        self.bias = nn.Parameter(torch.zeros(n))

    def branch(self, relation):
        return relation if self.separate else 0

    def query(self, source, target_category, relation=0):
        r = self.branch(relation)
        x = torch.cat(
            [self.encoder(source), self.encoder.category(target_category)], dim=-1
        )
        return F.normalize(self.queries[r](x), dim=-1)

    def candidate(self, target, relation=0):
        return F.normalize(
            self.candidates[self.branch(relation)](self.encoder(target)), dim=-1
        )

    def forward(self, source, target, category, relation=0):
        q = self.query(source, category, relation)
        c = self.candidate(target, relation)
        r = self.branch(relation)
        return (q * c).sum(-1) * self.log_scale[r].clamp(-2, 4.6).exp() + self.bias[r]


class PairTransformer(nn.Module):
    def __init__(self, encoder, dim=128):
        super().__init__()
        self.encoder = encoder
        self.relations = nn.Embedding(2, dim)
        self.roles = nn.Embedding(4, dim)
        self.category_projection = nn.Linear(32, dim)
        # Construct separately: TransformerEncoder clones identical initial weights otherwise.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim,
                    4,
                    dim_feedforward=256,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, source, target, category, relation=0):
        marker = self.relations.weight[relation].expand(source.shape[0], -1)
        x = torch.stack(
            [
                marker,
                self.encoder(source),
                self.encoder(target),
                self.category_projection(self.encoder.category(category)),
            ],
            dim=1,
        )
        x = x + self.roles.weight.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)  # Four feature tokens; no temporal/causal mask.
        return self.head(self.norm(x[:, 0])).squeeze(-1)
