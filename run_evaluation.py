"""
Steering Evaluation Script
===========================
Evaluates diversity steering on the TopKSAE-8192 model with ELSA-1024 backbone.

Produces results/steering_evaluation.json containing:
  1. Clean ELSA baseline metrics
  2. SAE passthrough (no steering) metrics
  3. Steering grid: N=[1,2,3,5] neurons × S=[0,1,2,...,100] strengths
  4. Interactive demo data for 3 sample users
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from tqdm import tqdm

from datasets import prepare_interaction_data, split_input_target_interactions, Dataloader
from elsa import ELSA
from sae import TopKSAE
from util import load_checkpoint, load_config_from_checkpoint

# ─── Configuration ───────────────────────────────────────────────────────────
ELSA_CKPT = "checkpoints/ML-25M/ELSA-1024-10977915.ckpt"
SAE_CKPT  = "checkpoints/ML-25M/TopKSAE-8192-d6337b64.ckpt"

NEURON_COUNTS = [1, 2, 3, 5]
STRENGTHS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100]

EVAL_USERS = 2000        # users for grid sweep metrics
CORR_USERS = 2000        # users for neuron correlation discovery
K = 20                   # top-K for Recall/nDCG/Precision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
elsa_cfg = load_config_from_checkpoint(ELSA_CKPT)
sae_cfg  = load_config_from_checkpoint(SAE_CKPT)

_, train_csr, val_csr, test_csr, _, _, _, items_map = prepare_interaction_data(elsa_cfg)

# Movie metadata
movies = {}
movies_path = os.path.join("data", "ML-25M", "movies.csv")
with open(movies_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        movies[row[0]] = {"title": row[1], "genres": row[2].replace("|", " | ")}

# Tag embeddings for diversity metric
print("Loading tag embeddings...")
tag_emb = {}
genome_path = os.path.join("data", "ML-25M", "genome-scores.csv")
raw_tags = {}
with open(genome_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        mid, tid, score = row[0], int(row[1]), float(row[2])
        if mid not in raw_tags:
            raw_tags[mid] = {}
        raw_tags[mid][tid] = score

max_tag_id = max(tid for tags in raw_tags.values() for tid in tags.keys())
for mid, tags in raw_tags.items():
    vec = np.zeros(max_tag_id + 1, dtype=np.float32)
    for tid, score in tags.items():
        vec[tid] = score
    tag_emb[mid] = vec
del raw_tags
print(f"  Loaded tag embeddings for {len(tag_emb)} movies ({max_tag_id+1} dims)")


# ─── Load models ─────────────────────────────────────────────────────────────
print("Loading models...")
n_items = items_map.shape[0]

elsa = ELSA(n_items, elsa_cfg["embedding_dim"]).to(device)
load_checkpoint(elsa, None, ELSA_CKPT, device)
elsa.eval()

sae_extra = {k: sae_cfg[k] for k in sae_cfg if k in ("l1_coef", "k")}
sae = TopKSAE(
    elsa_cfg["embedding_dim"],
    sae_cfg["embedding_dim"],
    sae_cfg["reconstruction_loss"],
    **sae_extra,
).to(device)
load_checkpoint(sae, None, SAE_CKPT, device)
sae.eval()

# ELSA item embeddings for CF diversity
elsa_item_emb = elsa.encoder.cpu().detach().numpy()
cf_emb = {str(items_map[i]): elsa_item_emb[i] for i in range(n_items)}


# ─── Diversity helpers ───────────────────────────────────────────────────────
def set_diversity(item_ids, emb_dict):
    """Average pairwise cosine distance of a set of items."""
    vecs = [emb_dict[str(i)] for i in item_ids if str(i) in emb_dict]
    if len(vecs) <= 1:
        return 0.0
    M = np.vstack(vecs)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    M = M / norms
    S = M @ M.T
    tri = 1.0 - S[np.triu_indices(len(vecs), k=1)]
    return float(np.mean(tri))


def user_diversity(user_csr_row, items_map, emb_dict):
    """Tag diversity of a user's interaction history."""
    item_ids = [items_map[idx] for idx in user_csr_row.indices]
    return set_diversity(item_ids, emb_dict)


# ─── Discover steering neurons via correlation ──────────────────────────────
print(f"Computing neuron-diversity correlations on {CORR_USERS} users...")
corr_csr = test_csr[:CORR_USERS]

# Compute diversity scores
div_scores = np.array([
    user_diversity(corr_csr[i], items_map, tag_emb)
    for i in tqdm(range(CORR_USERS), desc="User diversities")
])

# Get SAE activations
corr_batch = torch.tensor(corr_csr.toarray(), dtype=torch.float32, device=device)
with torch.no_grad():
    z = elsa.encode(corr_batch)
    e, _, _, _ = sae.encode(z.clone())   # clone to avoid in-place corruption
    activations = e.cpu().numpy()

# Pearson correlation per neuron
n_neurons = activations.shape[1]
correlations = np.zeros(n_neurons)
for i in range(n_neurons):
    a = activations[:, i]
    if np.std(a) > 0 and np.std(div_scores) > 0:
        correlations[i] = np.corrcoef(a, div_scores)[0, 1]

pos_neurons = np.argsort(correlations)[::-1]   # most positively correlated first
neg_neurons = np.argsort(correlations)          # most negatively correlated first

top_neuron = int(pos_neurons[0])
top_corr = float(correlations[top_neuron])
print(f"  Top diversity neuron: #{top_neuron} (r={top_corr:.4f})")
print(f"  Top 5: {[(int(pos_neurons[i]), round(float(correlations[pos_neurons[i]]),4)) for i in range(5)]}")


# ─── Recommendation helpers ─────────────────────────────────────────────────
@torch.no_grad()
def recommend_clean_elsa(batch, k=K):
    """Clean ELSA recommendations (no SAE)."""
    scores = nn.ReLU()(elsa.decode(elsa.encode(batch)) - batch)
    scores = torch.where(batch != 0, 0, scores)
    return torch.topk(scores, k)


@torch.no_grad()
def recommend_steered(batch, pos_n, neg_n, strength, k=K):
    """ELSA + SAE with optional steering."""
    z = elsa.encode(batch)
    e, _, x_mean, x_std = sae.encode(z.clone())

    # Apply steering
    if strength > 0:
        for idx in pos_n:
            e[:, idx] = strength
        for idx in neg_n:
            e[:, idx] = 0.0

    z_rec = sae.decode(e, x_mean, x_std)
    scores = nn.ReLU()(elsa.decode(z_rec) - batch)
    scores = torch.where(batch != 0, 0, scores)
    return torch.topk(scores, k)


# ─── Metric computation ─────────────────────────────────────────────────────
def compute_metrics(topk_indices, target_batch, input_batch, k=K):
    """Compute Recall@k, nDCG@k, Precision@k."""
    target_bool = target_batch.bool()

    # Recall
    predicted = torch.zeros_like(target_bool).scatter_(1, topk_indices, True)
    recall = (predicted & target_bool).sum(1).float() / torch.clamp(target_bool.sum(1).float(), min=1).clamp(max=k)

    # Precision
    precision = (predicted & target_bool).sum(1).float() / k

    # nDCG
    rel = target_bool.gather(1, topk_indices).float()
    gains = 2 ** rel - 1
    disc = torch.log2(torch.arange(2, k + 2, device=rel.device, dtype=torch.float))
    dcg = (gains / disc).sum(1)
    sorted_rel, _ = torch.sort(target_batch.float(), dim=1, descending=True)
    ideal_gains = 2 ** sorted_rel[:, :k] - 1
    idcg = (ideal_gains / disc).sum(1)
    idcg[idcg == 0] = 1
    ndcg = dcg / idcg

    return recall.cpu().numpy(), ndcg.cpu().numpy(), precision.cpu().numpy()


def compute_diversity_and_overlap(topk_indices, baseline_indices, items_map):
    """Compute tag div, cf div, and overlap with baseline for a batch."""
    tag_divs, cf_divs, overlaps = [], [], []
    for i in range(topk_indices.shape[0]):
        s_items = [items_map[idx] for idx in topk_indices[i].cpu().numpy()]
        tag_divs.append(set_diversity(s_items, tag_emb))
        cf_divs.append(set_diversity(s_items, cf_emb))
        if baseline_indices is not None:
            b_set = set(baseline_indices[i].cpu().numpy())
            s_set = set(topk_indices[i].cpu().numpy())
            overlaps.append(len(b_set & s_set) / K)
        else:
            overlaps.append(1.0)
    return np.mean(tag_divs), np.mean(cf_divs), np.mean(overlaps)


# ─── Prepare evaluation data ────────────────────────────────────────────────
print(f"\nPreparing evaluation on {EVAL_USERS} test users...")
eval_csr = test_csr[:EVAL_USERS]
inputs, targets = split_input_target_interactions(eval_csr, elsa_cfg["target_interaction_ratio"])
inputs_loader = Dataloader(inputs, 256, device)
targets_loader = Dataloader(targets, 256, device)

results = {
    "config": {
        "elsa_checkpoint": ELSA_CKPT,
        "sae_checkpoint": SAE_CKPT,
        "eval_users": EVAL_USERS,
        "k": K,
        "neuron_counts": NEURON_COUNTS,
        "strengths": STRENGTHS,
    },
    "top_neurons": {
        "positive": [int(x) for x in pos_neurons[:25]],
        "negative": [int(x) for x in neg_neurons[:25]],
        "correlations": {str(int(pos_neurons[i])): round(float(correlations[pos_neurons[i]]), 5) for i in range(25)},
    },
}


# ─── 1. Clean ELSA baseline ─────────────────────────────────────────────────
print("\n[1/3] Evaluating clean ELSA baseline...")
all_recall, all_ndcg, all_prec = [], [], []
all_baseline_indices = []
all_tag_div, all_cf_div = [], []

for inp_b, tgt_b in zip(inputs_loader, targets_loader):
    _, topk_idx = recommend_clean_elsa(inp_b)
    r, n, p = compute_metrics(topk_idx, tgt_b, inp_b)
    all_recall.extend(r); all_ndcg.extend(n); all_prec.extend(p)
    all_baseline_indices.append(topk_idx)
    td, cd, _ = compute_diversity_and_overlap(topk_idx, None, items_map)
    all_tag_div.append(td); all_cf_div.append(cd)

results["clean_elsa"] = {
    "recall": round(float(np.mean(all_recall)), 5),
    "ndcg": round(float(np.mean(all_ndcg)), 5),
    "precision": round(float(np.mean(all_prec)), 5),
    "tag_div": round(float(np.mean(all_tag_div)), 5),
    "cf_div": round(float(np.mean(all_cf_div)), 5),
}
print(f"  Clean ELSA → Recall={results['clean_elsa']['recall']:.4f}, nDCG={results['clean_elsa']['ndcg']:.4f}")

# Reconstruct baseline indices as one tensor for overlap
baseline_indices_all = torch.cat(all_baseline_indices, dim=0)


# ─── 2. SAE passthrough (strength=0) ────────────────────────────────────────
print("\n[2/3] Evaluating SAE passthrough (no steering)...")
all_recall, all_ndcg, all_prec = [], [], []
all_tag_div, all_cf_div, all_overlap = [], [], []
batch_idx = 0

for inp_b, tgt_b in zip(inputs_loader, targets_loader):
    bs = inp_b.shape[0]
    _, topk_idx = recommend_steered(inp_b, [], [], 0)
    r, n, p = compute_metrics(topk_idx, tgt_b, inp_b)
    all_recall.extend(r); all_ndcg.extend(n); all_prec.extend(p)
    bl_slice = baseline_indices_all[batch_idx:batch_idx+bs]
    td, cd, ol = compute_diversity_and_overlap(topk_idx, bl_slice, items_map)
    all_tag_div.append(td); all_cf_div.append(cd); all_overlap.append(ol)
    batch_idx += bs

results["sae_passthrough"] = {
    "recall": round(float(np.mean(all_recall)), 5),
    "ndcg": round(float(np.mean(all_ndcg)), 5),
    "precision": round(float(np.mean(all_prec)), 5),
    "tag_div": round(float(np.mean(all_tag_div)), 5),
    "cf_div": round(float(np.mean(all_cf_div)), 5),
    "overlap": round(float(np.mean(all_overlap)), 5),
}
print(f"  SAE pass  → Recall={results['sae_passthrough']['recall']:.4f}, nDCG={results['sae_passthrough']['ndcg']:.4f}")


# ─── 3. Steering grid sweep ─────────────────────────────────────────────────
print("\n[3/3] Steering grid sweep...")
results["steering"] = {}

for num_n in NEURON_COUNTS:
    pos_n = [int(x) for x in pos_neurons[:num_n]]
    neg_n = [int(x) for x in neg_neurons[:num_n]]
    results["steering"][str(num_n)] = {}

    print(f"\n  N={num_n} neurons (pos={pos_n})")
    for strength in STRENGTHS:
        all_recall, all_ndcg, all_prec = [], [], []
        all_tag_div, all_cf_div, all_overlap = [], [], []
        batch_idx = 0

        for inp_b, tgt_b in zip(inputs_loader, targets_loader):
            bs = inp_b.shape[0]
            _, topk_idx = recommend_steered(inp_b, pos_n, neg_n, float(strength))
            r, n, p = compute_metrics(topk_idx, tgt_b, inp_b)
            all_recall.extend(r); all_ndcg.extend(n); all_prec.extend(p)
            bl_slice = baseline_indices_all[batch_idx:batch_idx+bs]
            td, cd, ol = compute_diversity_and_overlap(topk_idx, bl_slice, items_map)
            all_tag_div.append(td); all_cf_div.append(cd); all_overlap.append(ol)
            batch_idx += bs

        entry = {
            "recall": round(float(np.mean(all_recall)), 5),
            "ndcg": round(float(np.mean(all_ndcg)), 5),
            "precision": round(float(np.mean(all_prec)), 5),
            "tag_div": round(float(np.mean(all_tag_div)), 5),
            "cf_div": round(float(np.mean(all_cf_div)), 5),
            "overlap": round(float(np.mean(all_overlap)), 5),
        }
        results["steering"][str(num_n)][str(strength)] = entry
        recall_pct = (entry["recall"] - results["clean_elsa"]["recall"]) / results["clean_elsa"]["recall"] * 100
        div_pct = (entry["tag_div"] - results["clean_elsa"]["tag_div"]) / results["clean_elsa"]["tag_div"] * 100
        print(f"    S={strength:3d}: Recall={entry['recall']:.4f} ({recall_pct:+.1f}%)  TagDiv={entry['tag_div']:.4f} ({div_pct:+.1f}%)  Overlap={entry['overlap']:.1%}")


# ─── 4. Interactive demo data for 3 users ────────────────────────────────────
print("\nGenerating interactive demo data...")
demo_users = [
    {"label": "User A (Low Diversity)", "idx": 216},
    {"label": "User B (Medium Diversity)", "idx": 55},
    {"label": "User C (High Diversity)", "idx": 165},
]

results["demo"] = {}
for u in demo_users:
    u_csr = test_csr[u["idx"]]
    u_batch = torch.tensor(u_csr.toarray(), dtype=torch.float32, device=device)

    # Baseline recommendations
    _, base_idx = recommend_clean_elsa(u_batch, k=10)
    base_items = [items_map[i] for i in base_idx[0].cpu().numpy()]

    user_data = {
        "baseline": {
            "recs": [{"title": movies.get(str(i), {}).get("title", "?"), "genres": movies.get(str(i), {}).get("genres", "?")} for i in base_items],
            "tag_div": round(set_diversity(base_items, tag_emb), 4),
            "cf_div": round(set_diversity(base_items, cf_emb), 4),
        },
        "steered": {},
    }

    for num_n in NEURON_COUNTS:
        pos_n = [int(x) for x in pos_neurons[:num_n]]
        neg_n = [int(x) for x in neg_neurons[:num_n]]
        user_data["steered"][str(num_n)] = {}

        for strength in STRENGTHS:
            _, s_idx = recommend_steered(u_batch, pos_n, neg_n, float(strength), k=10)
            s_items = [items_map[i] for i in s_idx[0].cpu().numpy()]
            overlap = len(set(base_items) & set(s_items)) / len(base_items)
            user_data["steered"][str(num_n)][str(strength)] = {
                "recs": [{"title": movies.get(str(i), {}).get("title", "?"), "genres": movies.get(str(i), {}).get("genres", "?")} for i in s_items],
                "tag_div": round(set_diversity(s_items, tag_emb), 4),
                "cf_div": round(set_diversity(s_items, cf_emb), 4),
                "overlap": round(overlap, 4),
            }

    results["demo"][u["label"]] = user_data

# ─── Save ────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "steering_evaluation.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {out_path}")
print("Done!")
