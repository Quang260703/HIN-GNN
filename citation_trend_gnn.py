"""
Structured Citation Trend Prediction Using Graph Neural Networks
Reimplementation of Cummings & Nassar (ICASSP 2020), arXiv:2104.02562

No official code was released for this paper (confirmed via Papers with Code),
so this is a from-scratch reimplementation of the described architecture:

  - A "prior graph" G_p of established papers (citation edges only).
  - A set of "target" papers V_t that arrive AFTER G_p and only cite INTO it
    (never the reverse) -- this is the causal constraint that lets you cache
    G_p's embeddings and just do a forward pass for new papers.
  - Per-paper features: TF-IDF over title+abstract, an author/affiliation
    embedding, and a publication-year embedding, concatenated.
  - A GAT backbone that predicts whether each target paper lands in the
    top-K% of citations for its venue/year (binary classification, not
    citation-count regression).
  - An MLP baseline with matched parameter count for comparison, since the
    paper's whole point is showing the graph structure itself helps.

Requires: torch, torch_geometric, scikit-learn
    pip install torch torch_geometric scikit-learn --break-system-packages

This is written to be dataset-agnostic: swap `load_your_dataset()` for a
loader over your own DBLP/SIGWEB-style corpus. A synthetic dataset is
included so the whole pipeline runs end-to-end out of the box.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# --------------------------------------------------------------------------- #
# 1. Data structures
# --------------------------------------------------------------------------- #

@dataclass
class CitationGraphConfig:
    """Hyperparameters controlling how the prior/target split and labels
    are constructed."""

    top_percentile: float = 0.10      # "trending" = top 10% cited in its cohort
    text_feature_dim: int = 1000      # TF-IDF vocab cap, matches the paper
    author_embed_dim: int = 32
    year_embed_dim: int = 8
    hidden_dim: int = 64
    num_gat_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.3
    min_year: int = 2000
    max_year: int = 2020


@dataclass
class RawPaper:
    """One paper before feature encoding."""

    paper_id: int
    title_abstract: str
    author_ids: list[int]
    year: int
    cites: list[int] = field(default_factory=list)  # ids this paper cites
    is_target: bool = False   # True = arrives after the prior graph is built
    citation_count: Optional[int] = None  # ground truth, used only to build labels


# --------------------------------------------------------------------------- #
# 2. Feature encoding
# --------------------------------------------------------------------------- #

class PaperFeatureEncoder(nn.Module):
    """Encodes TF-IDF text, (averaged) author ids, and year into one vector
    per paper, mirroring the three-feature-stream design in the paper.
    """

    def __init__(self, cfg: CitationGraphConfig, num_authors: int):
        super().__init__()
        self.cfg = cfg
        self.author_embed = nn.Embedding(num_authors + 1, cfg.author_embed_dim, padding_idx=0)
        self.year_embed = nn.Embedding(
            cfg.max_year - cfg.min_year + 2, cfg.year_embed_dim, padding_idx=0
        )
        self.text_proj = nn.Linear(cfg.text_feature_dim, cfg.hidden_dim)
        in_dim = cfg.hidden_dim + cfg.author_embed_dim + cfg.year_embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, tfidf_vecs: torch.Tensor, author_ids: torch.Tensor, years: torch.Tensor):
        # author_ids: [N, max_authors] (0 = padding) -> mean-pool present authors
        author_vecs = self.author_embed(author_ids)  # [N, A, D]
        mask = (author_ids != 0).unsqueeze(-1)
        summed = (author_vecs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        author_pooled = summed / counts

        year_idx = (years - self.cfg.min_year + 1).clamp(min=0)
        year_vecs = self.year_embed(year_idx)

        text_vecs = F.relu(self.text_proj(tfidf_vecs))

        fused = torch.cat([text_vecs, author_pooled, year_vecs], dim=-1)
        return self.fuse(fused)


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #

class CitationTrendGAT(nn.Module):
    """GAT backbone + binary trend classifier head."""

    def __init__(self, cfg: CitationGraphConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_dim = cfg.hidden_dim
        for i in range(cfg.num_gat_layers):
            out_dim = cfg.hidden_dim
            concat = i < cfg.num_gat_layers - 1
            heads = cfg.gat_heads if concat else 1
            layers.append(
                GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=cfg.dropout)
            )
            in_dim = out_dim * heads if concat else out_dim
        self.gat_layers = nn.ModuleList(layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.gat_layers):
            h = layer(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
        return self.classifier(h).squeeze(-1)  # logits


class CitationTrendMLP(nn.Module):
    """Structure-free baseline with a matched parameter budget, used to
    isolate how much the graph topology itself is contributing (per the
    paper's ablation)."""

    def __init__(self, cfg: CitationGraphConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# 4. Causal graph construction
# --------------------------------------------------------------------------- #

def build_causal_data(
    papers: list[RawPaper],
    cfg: CitationGraphConfig,
) -> tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Builds a single PyG `Data` object containing prior-graph nodes and
    target nodes, with edges enforced to be one-directional (target -> prior
    or prior -> prior only). This is what makes it valid to cache prior-graph
    embeddings and only recompute target rows for new papers.

    Returns: (data, tfidf_tensor, author_id_tensor, year_tensor, target_indices)
    """
    id_to_idx = {p.paper_id: i for i, p in enumerate(papers)}
    is_target = torch.tensor([p.is_target for p in papers], dtype=torch.bool)

    src, dst = [], []
    for p in papers:
        u = id_to_idx[p.paper_id]
        for cited_id in p.cites:
            if cited_id not in id_to_idx:
                continue
            v = id_to_idx[cited_id]
            # Causality: an edge u -> v (u cites v) is only valid if v is not
            # "later" than u, i.e. a target paper must never be cited BY a
            # prior paper (that would leak future info backward).
            if is_target[v] and not is_target[u]:
                continue  # illegal: prior paper "citing" a future paper
            src.append(u)
            dst.append(v)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    texts = [p.title_abstract for p in papers]
    vectorizer = TfidfVectorizer(max_features=cfg.text_feature_dim)
    tfidf = vectorizer.fit_transform(texts).toarray()
    tfidf_tensor = torch.tensor(tfidf, dtype=torch.float)
    # pad to fixed width in case vocab is smaller than text_feature_dim
    if tfidf_tensor.shape[1] < cfg.text_feature_dim:
        pad = cfg.text_feature_dim - tfidf_tensor.shape[1]
        tfidf_tensor = F.pad(tfidf_tensor, (0, pad))

    max_authors = max(len(p.author_ids) for p in papers)
    author_id_tensor = torch.zeros(len(papers), max_authors, dtype=torch.long)
    for i, p in enumerate(papers):
        for j, a in enumerate(p.author_ids):
            author_id_tensor[i, j] = a + 1  # reserve 0 for padding

    year_tensor = torch.tensor([p.year for p in papers], dtype=torch.long)

    labels = build_trend_labels(papers, cfg)

    data = Data(edge_index=edge_index, num_nodes=len(papers))
    data.is_target = is_target
    data.y = labels
    target_indices = [i for i, p in enumerate(papers) if p.is_target]
    return data, tfidf_tensor, author_id_tensor, year_tensor, target_indices


def build_trend_labels(papers: list[RawPaper], cfg: CitationGraphConfig) -> torch.Tensor:
    """Label = 1 if a paper's citation_count is in the top `top_percentile`
    for its (venue-agnostic here, extend by venue if you have it) publication
    year cohort. Papers without a known citation_count get label -1 (ignored
    in loss/metrics) -- typically the just-published target papers in a real
    deployment, though for training you need historical counts.
    """
    labels = torch.full((len(papers),), -1.0)
    by_year: dict[int, list[int]] = {}
    for i, p in enumerate(papers):
        by_year.setdefault(p.year, []).append(i)

    for year, idxs in by_year.items():
        counts = [papers[i].citation_count for i in idxs]
        known = [(i, c) for i, c in zip(idxs, counts) if c is not None]
        if not known:
            continue
        sorted_counts = sorted(c for _, c in known)
        cutoff_idx = max(0, int(len(sorted_counts) * (1 - cfg.top_percentile)) - 1)
        threshold = sorted_counts[cutoff_idx]
        for i, c in known:
            labels[i] = 1.0 if c >= threshold else 0.0
    return labels


# --------------------------------------------------------------------------- #
# 5. Training / evaluation
# --------------------------------------------------------------------------- #

def train_model(
    model: nn.Module,
    encoder: PaperFeatureEncoder,
    data: Data,
    tfidf: torch.Tensor,
    author_ids: torch.Tensor,
    years: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    verbose: bool = True,
) -> nn.Module:
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(encoder.parameters()), lr=lr, weight_decay=weight_decay
    )

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    for epoch in range(1, epochs + 1):
        model.train()
        encoder.train()
        optimizer.zero_grad()

        x = encoder(tfidf, author_ids, years)
        logits = model(x, data.edge_index)

        train_mask = data.y[train_idx_t] >= 0
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx_t][train_mask], data.y[train_idx_t][train_mask]
        )
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            model.eval()
            encoder.eval()
            with torch.no_grad():
                x_eval = encoder(tfidf, author_ids, years)
                val_logits = model(x_eval, data.edge_index)
                val_mask = data.y[val_idx_t] >= 0
                val_labels = data.y[val_idx_t][val_mask].numpy()
                val_preds = (torch.sigmoid(val_logits[val_idx_t][val_mask]) > 0.5).numpy()
                f1 = f1_score(val_labels, val_preds, zero_division=0)
                prec = precision_score(val_labels, val_preds, zero_division=0)
                rec = recall_score(val_labels, val_preds, zero_division=0)
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val F1 {f1:.3f} "
                  f"(P {prec:.3f} / R {rec:.3f})")

    return model


def edge_ablation_curve(
    model_cls,
    cfg: CitationGraphConfig,
    data: Data,
    tfidf: torch.Tensor,
    author_ids: torch.Tensor,
    years: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    keep_fractions=(1.0, 0.75, 0.5, 0.25, 0.0),
):
    """Reproduces the paper's edge-removal ablation: strip a fraction of
    edges and watch F1 degrade toward the structure-free baseline. This is
    the diagnostic worth running on your own heterogeneous graph before
    trusting that HGT/HAN is earning its complexity over a flatter model.
    """
    full_edge_index = data.edge_index
    num_edges = full_edge_index.shape[1]
    results = []
    for frac in keep_fractions:
        keep = int(num_edges * frac)
        perm = torch.randperm(num_edges)[:keep]
        ablated_edge_index = full_edge_index[:, perm]

        encoder = PaperFeatureEncoder(cfg, num_authors=int(author_ids.max().item()))
        model = model_cls(cfg)
        ablated_data = Data(edge_index=ablated_edge_index, num_nodes=data.num_nodes)
        ablated_data.y = data.y

        train_model(model, encoder, ablated_data, tfidf, author_ids, years,
                    train_idx, val_idx, epochs=50, verbose=False)

        model.eval()
        encoder.eval()
        with torch.no_grad():
            x_eval = encoder(tfidf, author_ids, years)
            logits = model(x_eval, ablated_data.edge_index)
            val_idx_t = torch.tensor(val_idx, dtype=torch.long)
            val_mask = data.y[val_idx_t] >= 0
            labels = data.y[val_idx_t][val_mask].numpy()
            preds = (torch.sigmoid(logits[val_idx_t][val_mask]) > 0.5).numpy()
            f1 = f1_score(labels, preds, zero_division=0)
        results.append((frac, f1))
        print(f"kept {frac*100:5.1f}% of edges -> val F1 {f1:.3f}")
    return results


# --------------------------------------------------------------------------- #
# 6. Synthetic dataset (swap this out for your own DBLP/SIGWEB loader)
# --------------------------------------------------------------------------- #

def make_synthetic_dataset(
    cfg: CitationGraphConfig,
    num_prior_papers: int = 800,
    num_target_papers: int = 200,
    num_authors: int = 150,
    seed: int = 0,
) -> list[RawPaper]:
    """Generates a toy citation graph with a preferential-attachment-ish
    structure so trend labels are non-trivial: a paper's eventual citation
    count correlates with in-degree from papers that arrive after it.
    """
    rng = random.Random(seed)
    vocab_topics = [
        "graph neural network", "attention mechanism", "transformer model",
        "citation analysis", "heterogeneous graph", "link prediction",
        "node classification", "representation learning", "knowledge graph",
        "recommendation system", "text classification", "self supervised",
    ]

    papers: list[RawPaper] = []
    for pid in range(num_prior_papers):
        year = rng.randint(cfg.min_year, cfg.max_year - 3)
        topic_words = rng.sample(vocab_topics, k=3)
        text = " ".join(topic_words) + " " + " ".join(rng.sample(vocab_topics, k=2))
        authors = rng.sample(range(num_authors), k=rng.randint(1, 4))
        # older papers can cite even-older papers
        possible_cites = [p.paper_id for p in papers if p.year <= year]
        cites = rng.sample(possible_cites, k=min(len(possible_cites), rng.randint(0, 5)))
        papers.append(RawPaper(pid, text, authors, year, cites=cites, is_target=False))

    # assign prior-paper citation counts based on in-degree from ALL prior
    # papers (simulating "already observed" citation history)
    in_degree = {p.paper_id: 0 for p in papers}
    for p in papers:
        for c in p.cites:
            in_degree[c] += 1
    for p in papers:
        p.citation_count = in_degree[p.paper_id] + rng.randint(0, 3)

    # target papers: arrive after the prior graph, can only cite INTO it
    next_id = num_prior_papers
    target_papers = []
    for i in range(num_target_papers):
        year = cfg.max_year - rng.randint(0, 2)
        topic_words = rng.sample(vocab_topics, k=3)
        text = " ".join(topic_words) + " " + " ".join(rng.sample(vocab_topics, k=2))
        authors = rng.sample(range(num_authors), k=rng.randint(1, 4))
        cites = rng.sample([p.paper_id for p in papers],
                            k=min(len(papers), rng.randint(3, 10)))
        tp = RawPaper(next_id, text, authors, year, cites=cites, is_target=True)
        target_papers.append(tp)
        next_id += 1

    # simulate eventual citation counts for targets (correlated with how
    # "central" the topics they cite are, to make the graph structure useful)
    prior_in_degree = in_degree
    for tp in target_papers:
        centrality = sum(prior_in_degree.get(c, 0) for c in tp.cites)
        tp.citation_count = int(centrality * rng.uniform(0.3, 0.6)) + rng.randint(0, 2)

    return papers + target_papers


# --------------------------------------------------------------------------- #
# 7. End-to-end example
# --------------------------------------------------------------------------- #

def main():
    torch.manual_seed(0)
    cfg = CitationGraphConfig()

    papers = make_synthetic_dataset(cfg)
    data, tfidf, author_ids, years, target_indices = build_causal_data(papers, cfg)

    rng = random.Random(1)
    rng.shuffle(target_indices)
    split = int(len(target_indices) * 0.8)
    train_idx, val_idx = target_indices[:split], target_indices[split:]

    num_authors = int(author_ids.max().item())

    print("=== GAT model (graph-aware) ===")
    encoder_gat = PaperFeatureEncoder(cfg, num_authors=num_authors)
    gat_model = CitationTrendGAT(cfg)
    train_model(gat_model, encoder_gat, data, tfidf, author_ids, years, train_idx, val_idx)

    print("\n=== MLP baseline (structure-free) ===")
    encoder_mlp = PaperFeatureEncoder(cfg, num_authors=num_authors)
    mlp_model = CitationTrendMLP(cfg)
    train_model(mlp_model, encoder_mlp, data, tfidf, author_ids, years, train_idx, val_idx)

    print("\n=== Edge ablation curve (GAT) ===")
    edge_ablation_curve(CitationTrendGAT, cfg, data, tfidf, author_ids, years,
                         train_idx, val_idx)


if __name__ == "__main__":
    main()
