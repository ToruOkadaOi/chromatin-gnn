#!/usr/bin/env python3
"""
Encode a new graph using a trained VGAE model.
Automatically infers hidden/latent dimensions from saved weights.
"""

import argparse, torch, numpy as np
from torch_geometric.nn import GCNConv, VGAE

# Reuse your Encoder definition directly here for clarity
class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, latent, dropout=0.2):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hidden)
        self.gc_mu = GCNConv(hidden, latent)
        self.gc_log = GCNConv(hidden, latent)
        self.dropout = dropout

    def forward(self, x, edge_index):
        import torch.nn.functional as F
        h = self.gc1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gc_mu(h, edge_index), self.gc_log(h, edge_index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # ---- Load data and model state ----
    data = torch.load(args.graph)
    model_state = torch.load(args.model, map_location="cpu")

    # ---- Infer dimensions dynamically ----
    in_dim = data.x.size(1)
    # detect hidden and latent dimensions safely
    keys = list(model_state.keys())
    gc1_weight = [k for k in keys if "gc1" in k and "weight" in k][0]
    gc_mu_weight = [k for k in keys if "gc_mu" in k and "weight" in k][0]

    hidden = model_state[gc1_weight].shape[0]
    latent = model_state[gc_mu_weight].shape[0]

    print(f"Inferred dims: in={in_dim}, hidden={hidden}, latent={latent}")

    enc = Encoder(in_dim=in_dim, hidden=hidden, latent=latent)
    model = VGAE(enc)
    model.load_state_dict(model_state)
    model.eval()

    # ---- Encode ----
    with torch.no_grad():
        z = model.encode(data.x.float(), data.edge_index)
    np.save(args.out, z.cpu().numpy())
    print(f"Saved embeddings → {args.out} shape={z.shape}")


if __name__ == "__main__":
    main()