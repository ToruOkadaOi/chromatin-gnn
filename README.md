# Graph learning for genome architecture

### workflow

```bash
# Build graph (contact matrix & bigwig needed)
python scripts/build_graph.py --mcool x.mcool --chrom chrx --res x
  --bigwigs xCTCFx.bw xH3K27me3x.bw --out data/chrx_xconditionx.pt

# Train model
python scripts/train_vgae.py --graph data/chrx_xconditionx.pt --epochs 100 --outdir results

# Encode treatment graph
python scripts/encode_graph.py --model results/model.pt --graph data/chrx_xtreatmentx.pt --out results/emb_xtreatmentx.npy

# Compare embeddings
python scripts/compare_embeddings_general.py --emb1 results/emb.npy --emb2 results/emb_xtreatmentx.npy \
  --label1 xControlx --label2 xTreatmentx --prefix results/chrx
```