# Baselines

MUSS is compared against the following selection methods (see the [README](README.md) for the full project overview).

| Method | Where | Notes |
|---|---|---|
| Random | `baselines.run_random_selection` | Uniform sampling |
| MMR | `baselines.run_experiment_MMR` | Greedy MMR over the full corpus |
| Clustering | `baselines.run_experiment_clustering` | FAISS k-means, top-quality per cluster |
| k-DPP | `baselines.run_experiment_DPP` | Exact k-DPP via `dppy` |
| DGDS | `baselines.run_DGDS` | Random partitioning + within-partition MMR (no cluster selection) |
| MUSS (full) | `helpers.run_MUSS` | Stage 1+2+3 with FAISS clustering |
| MUSS (rand. clusters) | `ablation_muss.run_MUSS_random_cluster_selection` | Stage 2 replaced by random selection |
| MUSS (rand. partition) | `ablation_muss.run_MUSS_rand_partition` | Stage 1 replaced by random partitioning |
