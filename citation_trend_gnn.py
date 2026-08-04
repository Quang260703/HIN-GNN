"""
Structured Citation Trend Prediction Using Graph Neural Networks
-- Heterogeneous (paper / author / venue) rewrite, loading directly from
   your DBLP-derived CSV export (papers, paper_references, paper_authors,
   conferences, cited_by).

*** Schema note ***
`papers` has no title/abstract column, so there is no text signal to build
a TF-IDF stream from. The paper's own feature vector here is just its
publication year (via conferences.year) plus two weak numeric signals:
log-scaled page count and whether a funding record is present. The real
predictive signal has to come from graph structure instead -- who wrote
it, what venue, what it cites -- which is honestly a fair test of whether
the GNN's structure is pulling weight, since there's no rich node feature
to fall back on.

*** Causal masking, recap ***
Citation edges are stored [cited, citing] so the citing paper is informed
by what it cites (not the reverse -- see the bug note from the previous
version). Author and venue nodes may inform ANY paper (prior or target)
but may only be INFORMED BY prior papers, so their embeddings stay a pure
"prior-graph" quantity computable once and reused, with no path for a
target paper's information to leak backward through a shared author or
venue.

Requires: torch, torch_geometric, scikit-learn, pandas
    pip install torch torch_geometric scikit-learn pandas --break-system-packages
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


# --------------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------------- #

@dataclass
class HeteroCitationConfig:
    top_percentile: float = 0.10
    hidden_dim: int = 64
    num_hgt_layers: int = 2
    hgt_heads: int = 4
    dropout: float = 0.3
    min_year: int = 1992
    max_year: int = 2015


# --------------------------------------------------------------------------- #
# 2. Raw data
# --------------------------------------------------------------------------- #

@dataclass
class RawPaper:
    paper_id: int
    author_ids: list[int]
    venue_id: Optional[int]
    year: int
    pages: float                 # 0.0 if unknown
    has_funding: bool
    cites: list[int] = field(default_factory=list)
    is_target: bool = False
    citation_count: Optional[int] = None   # from papers.cite_count


# --------------------------------------------------------------------------- #
# 3. Per-node-type feature encoders
# --------------------------------------------------------------------------- #

class PaperEncoder(nn.Module):
    """Year + weak numeric fields only -- no title/abstract available in
    this schema. Graph structure (author/venue/citation) is where the
    real signal has to come from."""

    def __init__(self, cfg: HeteroCitationConfig):
        super().__init__()
        self.cfg = cfg
        self.year_embed = nn.Embedding(cfg.max_year - cfg.min_year + 2, 16, padding_idx=0)
        self.numeric_proj = nn.Linear(2, 16)  # [log1p(pages), has_funding]
        self.fuse = nn.Sequential(
            nn.Linear(32, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, years: torch.Tensor, numeric_feats: torch.Tensor) -> torch.Tensor:
        year_idx = (years - self.cfg.min_year + 1).clamp(min=0)
        year_vecs = self.year_embed(year_idx)
        numeric_vecs = F.relu(self.numeric_proj(numeric_feats))
        return self.fuse(torch.cat([year_vecs, numeric_vecs], dim=-1))


class AuthorEncoder(nn.Module):
    """Author id embedding + two weak-but-safe covariates from authors_info:
    career start (year_first) and total publication count (pub_count).

    Deliberately EXCLUDED: authors_info.cite_count and authors_info.avg_cite.
    Both are aggregated across an author's whole career in the dataset --
    including their target-period papers -- so an author's avg_cite already
    partly reflects the citation outcome of the very paper you're trying to
    predict. Using them would leak the label through the author node, the
    same failure mode as feeding paper.cite_count into PaperEncoder.
    """

    def __init__(self, cfg: HeteroCitationConfig, num_authors: int):
        super().__init__()
        self.cfg = cfg
        self.id_embed = nn.Embedding(num_authors + 1, cfg.hidden_dim // 2, padding_idx=0)
        self.numeric_proj = nn.Linear(2, cfg.hidden_dim // 2)  # [year_first_norm, log1p(pub_count)]
        self.fuse = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, author_ids: torch.Tensor, numeric_feats: torch.Tensor) -> torch.Tensor:
        id_vecs = self.id_embed(author_ids)
        numeric_vecs = F.relu(self.numeric_proj(numeric_feats))
        return self.fuse(torch.cat([id_vecs, numeric_vecs], dim=-1))


class VenueEncoder(nn.Module):
    """Venue id embedding + a categorical publisher embedding. Publisher is
    set at publication time, so unlike citation-derived fields it carries
    no leakage risk."""

    def __init__(self, cfg: HeteroCitationConfig, num_venues: int, num_publishers: int):
        super().__init__()
        self.id_embed = nn.Embedding(num_venues + 1, cfg.hidden_dim // 2, padding_idx=0)
        self.publisher_embed = nn.Embedding(num_publishers + 1, cfg.hidden_dim // 2, padding_idx=0)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, venue_ids: torch.Tensor, publisher_ids: torch.Tensor) -> torch.Tensor:
        id_vecs = self.id_embed(venue_ids)
        pub_vecs = self.publisher_embed(publisher_ids)
        return self.fuse(torch.cat([id_vecs, pub_vecs], dim=-1))


class HeteroEncoders(nn.Module):
    def __init__(self, cfg: HeteroCitationConfig, num_authors: int, num_venues: int, num_publishers: int):
        super().__init__()
        self.paper_encoder = PaperEncoder(cfg)
        self.author_encoder = AuthorEncoder(cfg, num_authors)
        self.venue_encoder = VenueEncoder(cfg, num_venues, num_publishers)

    def forward(self, raw: dict) -> dict:
        return {
            "paper": self.paper_encoder(raw["paper_years"], raw["paper_numeric"]),
            "author": self.author_encoder(raw["author_ids"], raw["author_numeric"]),
            "venue": self.venue_encoder(raw["venue_ids"], raw["venue_publisher_ids"]),
        }


# --------------------------------------------------------------------------- #
# 4. Model
# --------------------------------------------------------------------------- #

class CitationTrendHGT(nn.Module):
    def __init__(self, cfg: HeteroCitationConfig, metadata):
        super().__init__()
        self.layers = nn.ModuleList([
            HGTConv(cfg.hidden_dim, cfg.hidden_dim, metadata, heads=cfg.hgt_heads)
            for _ in range(cfg.num_hgt_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h_dict = x_dict
        for i, layer in enumerate(self.layers):
            h_dict = layer(h_dict, edge_index_dict)
            if i < len(self.layers) - 1:
                h_dict = {k: F.elu(self.dropout(v)) for k, v in h_dict.items()}
        return self.classifier(h_dict["paper"]).squeeze(-1)


# --------------------------------------------------------------------------- #
# 5. Heterogeneous causal graph construction
# --------------------------------------------------------------------------- #

def build_hetero_causal_data(
    papers: list[RawPaper],
    cfg: HeteroCitationConfig,
    author_info: Optional[dict[int, dict]] = None,
    venue_publisher: Optional[dict[int, str]] = None,
):
    """
    author_info: {author_id: {"year_first": int, "pub_count": int}}
        (cite_count / avg_cite intentionally not consumed here -- see
        AuthorEncoder docstring)
    venue_publisher: {conf_id: publisher_str}
    """
    author_info = author_info or {}
    venue_publisher = venue_publisher or {}

    paper_ids = [p.paper_id for p in papers]
    pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    is_target = torch.tensor([p.is_target for p in papers], dtype=torch.bool)

    author_ids = sorted({a for p in papers for a in p.author_ids})
    aid_to_idx = {a: i for i, a in enumerate(author_ids)}

    venue_ids = sorted({p.venue_id for p in papers if p.venue_id is not None})
    vid_to_idx = {v: i for i, v in enumerate(venue_ids)}

    # paper -cites-> paper : [cited, citing], causally masked
    cite_src, cite_dst = [], []
    for p in papers:
        citing_idx = pid_to_idx[p.paper_id]
        for cited_id in p.cites:
            if cited_id not in pid_to_idx:
                continue  # cited paper outside our loaded year range -- skip
            cited_idx = pid_to_idx[cited_id]
            if is_target[cited_idx] and not is_target[citing_idx]:
                continue
            cite_src.append(cited_idx)
            cite_dst.append(citing_idx)

    # author -writes-> paper : all papers.  paper -has_author-> author : prior only
    writes_src, writes_dst = [], []
    has_author_src, has_author_dst = [], []
    for p in papers:
        p_idx = pid_to_idx[p.paper_id]
        for a in p.author_ids:
            a_idx = aid_to_idx[a]
            writes_src.append(a_idx)
            writes_dst.append(p_idx)
            if not p.is_target:
                has_author_src.append(p_idx)
                has_author_dst.append(a_idx)

    # venue -publishes-> paper : all papers.  paper -in_venue-> venue : prior only
    publishes_src, publishes_dst = [], []
    in_venue_src, in_venue_dst = [], []
    for p in papers:
        if p.venue_id is None:
            continue
        p_idx = pid_to_idx[p.paper_id]
        v_idx = vid_to_idx[p.venue_id]
        publishes_src.append(v_idx)
        publishes_dst.append(p_idx)
        if not p.is_target:
            in_venue_src.append(p_idx)
            in_venue_dst.append(v_idx)

    data = HeteroData()
    data["paper"].num_nodes = len(papers)
    data["author"].num_nodes = len(author_ids)
    data["venue"].num_nodes = len(venue_ids)

    data["paper", "cites", "paper"].edge_index = torch.tensor([cite_src, cite_dst], dtype=torch.long)
    data["author", "writes", "paper"].edge_index = torch.tensor([writes_src, writes_dst], dtype=torch.long)
    data["paper", "has_author", "author"].edge_index = torch.tensor([has_author_src, has_author_dst], dtype=torch.long)
    data["venue", "publishes", "paper"].edge_index = torch.tensor([publishes_src, publishes_dst], dtype=torch.long)
    data["paper", "in_venue", "venue"].edge_index = torch.tensor([in_venue_src, in_venue_dst], dtype=torch.long)

    years = torch.tensor([p.year for p in papers], dtype=torch.long)
    numeric = torch.tensor(
        [[torch.log1p(torch.tensor(float(p.pages))).item(), float(p.has_funding)] for p in papers],
        dtype=torch.float,
    )
    author_id_tensor = torch.tensor([aid_to_idx[a] + 1 for a in author_ids], dtype=torch.long)
    venue_id_tensor = torch.tensor([vid_to_idx[v] + 1 for v in venue_ids], dtype=torch.long)

    # --- author covariates: year_first (normalized), pub_count (log1p) ---
    year_span = max(cfg.max_year - cfg.min_year, 1)
    author_numeric_rows = []
    for a in author_ids:
        info = author_info.get(a, {})
        year_first = info.get("year_first", cfg.min_year)
        pub_count = info.get("pub_count", 0)
        norm_year = (year_first - cfg.min_year) / year_span
        author_numeric_rows.append([norm_year, float(pub_count)])
    author_numeric = torch.tensor(author_numeric_rows, dtype=torch.float)
    author_numeric[:, 1] = torch.log1p(author_numeric[:, 1])

    # --- venue publisher: categorical vocab, index 0 reserved for unknown ---
    publisher_vocab: dict[str, int] = {}
    venue_publisher_ids = []
    for v in venue_ids:
        pub = venue_publisher.get(v)
        if pub is None or (isinstance(pub, float) and pub != pub):  # NaN check
            venue_publisher_ids.append(0)
            continue
        if pub not in publisher_vocab:
            publisher_vocab[pub] = len(publisher_vocab) + 1
        venue_publisher_ids.append(publisher_vocab[pub])
    venue_publisher_tensor = torch.tensor(venue_publisher_ids, dtype=torch.long)
    num_publishers = len(publisher_vocab)

    labels = build_trend_labels(papers, cfg)
    data["paper"].y = labels
    data["paper"].is_target = is_target

    target_indices = [i for i, p in enumerate(papers) if p.is_target]

    raw_features = {
        "paper_years": years,
        "paper_numeric": numeric,
        "author_ids": author_id_tensor,
        "author_numeric": author_numeric,
        "venue_ids": venue_id_tensor,
        "venue_publisher_ids": venue_publisher_tensor,
    }
    id_maps = {
        "pid_to_idx": pid_to_idx,
        "aid_to_idx": aid_to_idx,
        "vid_to_idx": vid_to_idx,
        "num_publishers": num_publishers,
    }
    return data, raw_features, id_maps, target_indices


def build_trend_labels(papers: list[RawPaper], cfg: HeteroCitationConfig) -> torch.Tensor:
    labels = torch.full((len(papers),), -1.0)
    by_year: dict[int, list[int]] = {}
    for i, p in enumerate(papers):
        by_year.setdefault(p.year, []).append(i)
    for year, idxs in by_year.items():
        known = [(i, papers[i].citation_count) for i in idxs if papers[i].citation_count is not None]
        if not known:
            continue
        sorted_counts = sorted(c for _, c in known)
        cutoff_idx = max(0, int(len(sorted_counts) * (1 - cfg.top_percentile)) - 1)
        threshold = sorted_counts[cutoff_idx]
        for i, c in known:
            labels[i] = 1.0 if c >= threshold else 0.0
    return labels


# --------------------------------------------------------------------------- #
# 6. Training
# --------------------------------------------------------------------------- #

def train_hetero_model(
    model: nn.Module,
    encoders: HeteroEncoders,
    data: HeteroData,
    raw_features: dict,
    train_idx: list[int],
    val_idx: list[int],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 15,
    verbose: bool = True,
) -> nn.Module:
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(encoders.parameters()), lr=lr, weight_decay=weight_decay
    )
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    best_f1, best_state, no_improve = -1.0, None, 0
    history = {"epoch": [], "loss": [], "val_f1": [], "val_precision": [], "val_recall": []}

    for epoch in range(1, epochs + 1):
        model.train()
        encoders.train()
        optimizer.zero_grad()

        x_dict = encoders(raw_features)
        logits = model(x_dict, data.edge_index_dict)

        train_mask = data["paper"].y[train_idx_t] >= 0
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx_t][train_mask], data["paper"].y[train_idx_t][train_mask]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        encoders.eval()
        with torch.no_grad():
            x_eval = encoders(raw_features)
            val_logits = model(x_eval, data.edge_index_dict)
            val_mask = data["paper"].y[val_idx_t] >= 0
            val_labels = data["paper"].y[val_idx_t][val_mask].numpy()
            val_preds = (torch.sigmoid(val_logits[val_idx_t][val_mask]) > 0.5).numpy()
            f1 = f1_score(val_labels, val_preds, zero_division=0)
            prec = precision_score(val_labels, val_preds, zero_division=0)
            rec = recall_score(val_labels, val_preds, zero_division=0)

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["val_f1"].append(f1)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "encoders": {k: v.clone() for k, v in encoders.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val F1 {f1:.3f} "
                  f"(P {prec:.3f} / R {rec:.3f}) | best F1 {best_f1:.3f}")

        if no_improve >= patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        encoders.load_state_dict(best_state["encoders"])
    return model, history


def relation_ablation(cfg, metadata, data, raw_features, num_authors, num_venues, num_publishers,
                       train_idx, val_idx, epochs: int = 40):
    """Drops one relation type at a time to see which structure (citation
    vs. authorship vs. venue) is actually carrying the signal."""
    relations = list(data.edge_index_dict.keys())
    print(f"\nFull graph relations: {relations}")

    for relation_to_drop in relations + [None]:
        ablated = HeteroData()
        for ntype in data.node_types:
            ablated[ntype].num_nodes = data[ntype].num_nodes
        ablated["paper"].y = data["paper"].y
        for rel, edge_index in data.edge_index_dict.items():
            ablated[rel].edge_index = (
                torch.empty((2, 0), dtype=torch.long) if rel == relation_to_drop else edge_index
            )

        encoders = HeteroEncoders(cfg, num_authors, num_venues, num_publishers)
        model = CitationTrendHGT(cfg, metadata)
        model, _ = train_hetero_model(model, encoders, ablated, raw_features, train_idx, val_idx,
                                       epochs=epochs, verbose=False)

        model.eval()
        encoders.eval()
        with torch.no_grad():
            x_eval = encoders(raw_features)
            logits = model(x_eval, ablated.edge_index_dict)
            val_idx_t = torch.tensor(val_idx, dtype=torch.long)
            val_mask = data["paper"].y[val_idx_t] >= 0
            labels = data["paper"].y[val_idx_t][val_mask].numpy()
            preds = (torch.sigmoid(logits[val_idx_t][val_mask]) > 0.5).numpy()
            f1 = f1_score(labels, preds, zero_division=0)

        tag = "FULL GRAPH (nothing dropped)" if relation_to_drop is None else f"dropped {relation_to_drop}"
        print(f"{tag:55s} -> val F1 {f1:.3f}")


# --------------------------------------------------------------------------- #
# 7. Real data loader -- your CSV schema
# --------------------------------------------------------------------------- #

def load_dblp_from_csv(
    papers_csv: str,
    paper_references_csv: str,
    paper_authors_csv: str,
    conferences_csv: str,
    cited_by_csv: Optional[str] = None,
    authors_info_csv: Optional[str] = None,
    prior_year_cutoff: int = 2010,
    min_year: int = 1992,
    max_year: int = 2015,
):
    """Loads your exact schema:
        papers(dblp_key, crossref, doi, paper_id, cite_count, pages, conf_id, funding)
        paper_references(paper_id, refer_id)      -- paper_id cites refer_id
        paper_authors(paper_id, author_id, affiliation)
        conferences(dblp_key, year, publisher, title, doi, conf_id)
        cited_by(paper_id, cite_id)                -- cite_id cites paper_id (optional, adds coverage)
        authors_info(author_id, author_name, year_first, year_last, pub_count,
                     cite_count, avg_cite)          -- optional; cite_count/avg_cite
                     are read but NOT returned as usable features (leakage -- see
                     AuthorEncoder docstring)

    Returns: (papers, author_info, venue_publisher)
        author_info: {author_id: {"year_first": int, "pub_count": int}}
        venue_publisher: {conf_id: publisher_str}
    """
    conferences = pd.read_csv(conferences_csv)
    print("Conferences:", len(conferences))
    conf_year = conferences.dropna(subset=["conf_id"]).set_index("conf_id")["year"].to_dict()
    venue_publisher = conferences.dropna(subset=["conf_id"]).set_index("conf_id")["publisher"].to_dict()

    papers_df = pd.read_csv(papers_csv)
    print("Papers:", len(papers_df))
    papers_df = papers_df[papers_df["conf_id"].isin(conf_year.keys())].copy()
    papers_df["year"] = papers_df["conf_id"].map(conf_year)
    papers_df = papers_df[(papers_df["year"] >= min_year) & (papers_df["year"] <= max_year)]

    authors_df = pd.read_csv(paper_authors_csv)
    authors_by_paper: dict[int, list[int]] = (
        authors_df.groupby("paper_id")["author_id"].apply(list).to_dict()
    )

    refs_df = pd.read_csv(paper_references_csv)
    cites_by_paper: dict[int, set[int]] = {}
    for paper_id, refer_id in zip(refs_df["paper_id"], refs_df["refer_id"]):
        cites_by_paper.setdefault(paper_id, set()).add(refer_id)

    if cited_by_csv:
        cb_df = pd.read_csv(cited_by_csv)
        # cited_by: paper_id is cited by cite_id  =>  cite_id cites paper_id
        for paper_id, cite_id in zip(cb_df["paper_id"], cb_df["cite_id"]):
            cites_by_paper.setdefault(cite_id, set()).add(paper_id)

    papers: list[RawPaper] = []
    for row in papers_df.itertuples(index=False):
        pid = row.paper_id
        pages = float(row.pages) if pd.notna(row.pages) else 0.0
        has_funding = pd.notna(row.funding)
        papers.append(RawPaper(
            paper_id=pid,
            author_ids=authors_by_paper.get(pid, []),
            venue_id=row.conf_id if pd.notna(row.conf_id) else None,
            year=int(row.year),
            pages=pages,
            has_funding=has_funding,
            cites=list(cites_by_paper.get(pid, [])),
            is_target=int(row.year) > prior_year_cutoff,
            citation_count=int(row.cite_count) if pd.notna(row.cite_count) else None,
        ))

    author_info: dict[int, dict] = {}
    if authors_info_csv:
        ai_df = pd.read_csv(authors_info_csv)
        for row in ai_df.itertuples(index=False):
            # NOTE: row.cite_count / row.avg_cite are intentionally not
            # copied into author_info -- see AuthorEncoder docstring.
            author_info[row.author_id] = {
                "year_first": int(row.year_first) if pd.notna(row.year_first) else min_year,
                "pub_count": int(row.pub_count) if pd.notna(row.pub_count) else 0,
            }
    print("Matching conf_ids:",
      papers_df["conf_id"].isin(conf_year.keys()).sum())

    papers_df["year"] = papers_df["conf_id"].map(conf_year)

    print("Mapped years:", papers_df["year"].notna().sum())
    print("papers dtype:", papers_df["conf_id"].dtype)
    print("conference dtype:", conferences["conf_id"].dtype)

    print(conferences.columns.tolist())
    print(conferences.head(3).to_string())
    print(conferences.dtypes)
    return papers, author_info, venue_publisher


# --------------------------------------------------------------------------- #
# 7b. Plotting
# --------------------------------------------------------------------------- #

def plot_training_curves(history: dict, out_path: str = "training_curves.png"):
    """Plots validation F1/precision/recall and training loss over epochs,
    side by side. `history` is the dict returned by train_hetero_model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history["epoch"], history["val_f1"], label="F1", linewidth=2, color="#1f77b4")
    ax1.plot(history["epoch"], history["val_precision"], label="Precision",
              linestyle="--", color="#ff7f0e")
    ax1.plot(history["epoch"], history["val_recall"], label="Recall",
              linestyle="--", color="#2ca02c")
    best_epoch = history["epoch"][int(torch.tensor(history["val_f1"]).argmax())]
    best_f1 = max(history["val_f1"])
    ax1.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7,
                label=f"best F1 ({best_f1:.3f}) @ epoch {best_epoch}")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("score")
    ax1.set_title("Validation F1 / Precision / Recall")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(history["epoch"], history["loss"], color="#d62728", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training loss (BCE)")
    ax2.set_title("Training Loss")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved training curves to {out_path}")


# --------------------------------------------------------------------------- #
# 8. End-to-end example
# --------------------------------------------------------------------------- #

def main():
    torch.manual_seed(0)
    cfg = HeteroCitationConfig()

    papers, author_info, venue_publisher = load_dblp_from_csv(
        papers_csv="papers.csv",
        paper_references_csv="paper_references.csv",
        paper_authors_csv="paper_authors.csv",
        conferences_csv="conferences.csv",
        cited_by_csv="cited_by.csv",           # optional -- pass None to skip
        authors_info_csv="authors_info.csv",   # optional -- pass None to skip
        prior_year_cutoff=2010,
    )
    print(f"Loaded {len(papers)} papers "
          f"({sum(p.is_target for p in papers)} target / {sum(not p.is_target for p in papers)} prior)")

    data, raw_features, id_maps, target_indices = build_hetero_causal_data(
        papers, cfg, author_info=author_info, venue_publisher=venue_publisher
    )
    metadata = data.metadata()

    rng = random.Random(1)
    rng.shuffle(target_indices)
    split = int(len(target_indices) * 0.8)
    train_idx, val_idx = target_indices[:split], target_indices[split:]

    num_authors = len(id_maps["aid_to_idx"])
    num_venues = len(id_maps["vid_to_idx"])
    num_publishers = id_maps["num_publishers"]

    print("\n=== HGT model (heterogeneous, graph-aware) ===")
    encoders = HeteroEncoders(cfg, num_authors, num_venues, num_publishers)
    model = CitationTrendHGT(cfg, metadata)
    model, history = train_hetero_model(model, encoders, data, raw_features, train_idx, val_idx)

    plot_training_curves(history, out_path="training_curves.png")

    print("\n=== Per-relation ablation (HGT) ===")
    relation_ablation(cfg, metadata, data, raw_features, num_authors, num_venues, num_publishers,
                       train_idx, val_idx)


if __name__ == "__main__":
    main()

    def main():
        torch.manual_seed(0)
    cfg = HeteroCitationConfig()

    papers, author_info, venue_publisher = load_dblp_from_csv(
        papers_csv="papers.csv",
        paper_references_csv="paper_references.csv",
        paper_authors_csv="paper_authors.csv",
        conferences_csv="conferences.csv",
        cited_by_csv="cited_by.csv",           # optional -- pass None to skip
        authors_info_csv="authors_info.csv",   # optional -- pass None to skip
        prior_year_cutoff=2010,
    )
    print(f"Loaded {len(papers)} papers "
            f"({sum(p.is_target for p in papers)} target / {sum(not p.is_target for p in papers)} prior)")

    data, raw_features, id_maps, target_indices = build_hetero_causal_data(
        papers, cfg, author_info=author_info, venue_publisher=venue_publisher
    )
    metadata = data.metadata()

    rng = random.Random(1)
    rng.shuffle(target_indices)
    split = int(len(target_indices) * 0.8)
    train_idx, val_idx = target_indices[:split], target_indices[split:]

    num_authors = len(id_maps["aid_to_idx"])
    num_venues = len(id_maps["vid_to_idx"])
    num_publishers = id_maps["num_publishers"]

    print("\n=== HGT model (heterogeneous, graph-aware) ===")
    encoders = HeteroEncoders(cfg, num_authors, num_venues, num_publishers)
    model = CitationTrendHGT(cfg, metadata)
    model, history = train_hetero_model(model, encoders, data, raw_features, train_idx, val_idx)

    plot_training_curves(history, out_path="training_curves.png")

    print("\n=== Per-relation ablation (HGT) ===")
    relation_ablation(cfg, metadata, data, raw_features, num_authors, num_venues, num_publishers,
                        train_idx, val_idx)


    if __name__ == "__main__":
        main()