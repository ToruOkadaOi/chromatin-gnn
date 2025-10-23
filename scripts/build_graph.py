#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
import cooler
import pyBigWig
from torch_geometric.data import Data


def bin_bigwig(bw_path, chrom, bins):
    """Average bigWig signal across each genomic bin"""
    bw = pyBigWig.open(bw_path)
    if chrom not in bw.chroms():
        raise ValueError(f"{chrom} not found in {bw_path}. Available: {list(bw.chroms().keys())[:5]}...")
    chrom_len = bw.chroms(chrom)
    vals = []
    for s, e in bins:
        s = max(0, s)
        e = min(chrom_len, e)
        if s >= e:
            vals.append(0.0)
            continue
        v = bw.stats(chrom, s, e, type="mean")[0]
        vals.append(0.0 if v is None or np.isnan(v) else v)
    bw.close()
    return np.array(vals)


def build_graph(mcool_path, chrom, res, bigwigs, out_path, max_dist=5_000_000):
    """Convert .mcool + bigWigs to PyTorch Geometric Data object."""
    print(f"Processing {chrom} at {res} bp resolution...")

    # Load pixels
    c = cooler.Cooler(f"{mcool_path}::resolutions/{res}")
    pixels = c.matrix(balance=True, as_pixels=True, join=True).fetch(chrom)
    pixels = pixels.query(f"chrom1 == chrom2 and abs(start2 - start1) <= {max_dist}")

    # Map genomic coordinates to bin IDs
    bins_df = c.bins().fetch(chrom)
    bins_df["bin_id"] = np.arange(len(bins_df))
    start_to_bin = dict(zip(bins_df["start"].values, bins_df["bin_id"].values))

    valid = pixels["start1"].isin(start_to_bin) & pixels["start2"].isin(start_to_bin)
    pixels = pixels.loc[valid]

    bin1 = pixels["start1"].map(start_to_bin).values
    bin2 = pixels["start2"].map(start_to_bin).values
    edge_index = torch.tensor([bin1, bin2], dtype=torch.long)

    # Edge weights
    if "balanced" in pixels.columns and pixels["balanced"].notna().any():
        w = pixels["balanced"].fillna(0).values
    else:
        w = pixels["count"].values
    edge_weight = torch.tensor(np.log1p(w), dtype=torch.float)

    # Node features
    starts = bins_df["start"].values
    bins = [(int(s), int(s + res)) for s in starts]
    node_feats = []
    for bw in bigwigs:
        print(f"  Adding feature from {bw}")
        node_feats.append(bin_bigwig(bw, chrom, bins))
    x = torch.tensor(np.stack(node_feats, axis=1), dtype=torch.float)

    # Save graph
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    torch.save(data, out_path)
    print(f"Saved {chrom}: {x.shape[0]} nodes, {edge_index.shape[1]} edges → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build graph from Micro-C and bigWigs")
    p.add_argument("--mcool", required=True, help="Path to .mcool file")
    p.add_argument("--chrom", required=True, help="Chromosome name (e.g., chr21)")
    p.add_argument("--res", type=int, default=10000, help="Resolution (bp)")
    p.add_argument("--bigwigs", nargs="+", required=True, help="List of bigWig feature files")
    p.add_argument("--out", required=True, help="Output .pt file path")
    p.add_argument("--max_dist", type=int, default=5_000_000, help="Max genomic distance for edges")
    args = p.parse_args()

    build_graph(args.mcool, args.chrom, args.res, args.bigwigs, args.out, args.max_dist)
