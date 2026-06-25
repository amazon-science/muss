from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from muss.src.candidate_retrieval.ablation_muss import (
    run_MUSS_random_cluster_selection,
    run_MUSS_rand_partition
)
from muss.src.candidate_retrieval.helpers import (
    run_MUSS,
    save_result
)

MMR_DIVERSITY_TYPE_SUM = "sum"
MMR_DIVERSITY_TYPE_MIN = "min"

# store the result
output_df = pd.DataFrame()

# Replace this block with a `pd.read_parquet(...)` of your own candidate
# corpus. The DataFrame must have columns: emb (np.ndarray per row),
# quality_score (float), label (int), subcategory_code (str), ctr (float).
def make_synthetic_corpus(n=5000, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return pd.DataFrame({
        "emb":              list(emb),
        "quality_score":    rng.random(n),
        "label":            rng.integers(0, 2, size=n),
        "subcategory_code": rng.integers(0, 50, size=n).astype(str),
        "ctr":              rng.random(n) * 0.3,
    })

selected_df = make_synthetic_corpus()

dataset_name = 'synthetic'

print("dataset_name", dataset_name, selected_df.shape)

# number of selected items
k = 500

# run the main MUSS algorithm ===============================================================
for rand_val in tqdm([1,2,3,4,5]):
    for lamb in tqdm([0.1,0.3,0.5,0.7,0.9]):
        for lamb_c in [0.1,0.3,0.5,0.7,0.9]:
    
            for m_value in [50,100,200]:
                for n_partition_value in [100,200,500]:
                    
                    if n_partition_value > selected_df.shape[0]/5:
                        continue
                    if m_value >= n_partition_value:
                        continue
    
                    eval_output, _ = run_MUSS(selected_df, k = k, k_within = 50, m = m_value, 
                              n_partitions = n_partition_value, n_jobs=25, lamb = lamb, 
                                              lamb_c = lamb_c, dataset = dataset_name,
                                              random_state=rand_val, div_type=MMR_DIVERSITY_TYPE_SUM)
    
                    output_df = pd.concat ([output_df,eval_output]).drop_duplicates().reset_index(drop=True)
save_result(output_df)

for rand_val in tqdm([1,2,3,4,5]):
    for lamb in tqdm([0.1,0.3,0.5,0.7,0.9]):
        for lamb_c in [0.1,0.3,0.5,0.7,0.9]:
    
            for m_value in [50,100,200]:
                for n_partition_value in [100,200,500]:
                    
                    if n_partition_value > selected_df.shape[0]/5:
                        continue
    
                    if m_value >= n_partition_value:
                        continue
    
                    eval_output, _ = run_MUSS(selected_df, k = k, k_within = 50, m = m_value, 
                              n_partitions = n_partition_value, n_jobs=25, lamb = lamb, 
                                              lamb_c = lamb_c, dataset = dataset_name,
                                           random_state=rand_val, div_type=MMR_DIVERSITY_TYPE_MIN)
    
                    output_df = pd.concat ([output_df,eval_output]).drop_duplicates().reset_index(drop=True)
    
save_result(output_df)


# Ablation Study with Random partition=============================
for rand_val in tqdm([1,2,3,4,5]):
    for lamb in [0.1,0.3,0.5,0.7,0.9]:
        for lamb_c in [0.1,0.3,0.5,0.7,0.9]:
    
            for m_value in [50,100,200]:
                for n_partition_value in [100,200,500]:  
    
                    if n_partition_value > selected_df.shape[0]/5:
                        continue
                    if selected_df.shape[0]/n_partition_value<=30:
                        continue
    
                    eval_output,_ = run_MUSS_rand_partition(selected_df, k = k, k_within = 50, m = m_value, 
                                       n_partitions = n_partition_value, lamb = lamb, lamb_c=lamb_c,
                                          n_jobs=25, random_state=rand_val,    dataset = dataset_name)
                    output_df = pd.concat ([output_df,eval_output]).drop_duplicates().reset_index(drop=True)
    
save_result(output_df)


# Ablation Study with Random cluster selection ========================================
for rand_val in tqdm([1,2,3,4,5]): # over 5 random seeds
    for lamb in tqdm([0.1,0.3,0.5,0.7,0.9]):
        for m_value in [50, 100, 200]:
            for n_partition_value in [100,200,500]:        
    
                if n_partition_value > selected_df.shape[0]/5:
                    continue

                if selected_df.shape[0]/n_partition_value<=30:
                    continue
    
                eval_output, _ = run_MUSS_random_cluster_selection(selected_df, k = k, k_within=50, 
                                   m = m_value, n_partitions = n_partition_value,  lamb = lamb, \
                                   n_jobs=25,  random_state=rand_val,  dataset = dataset_name)
                output_df = pd.concat ([output_df,eval_output]).drop_duplicates().reset_index(drop=True)
    
save_result(output_df)