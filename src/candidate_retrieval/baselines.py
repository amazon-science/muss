from .helpers import process_feature_quality, evaluate, QUALITY_SCORE
import faiss
import numpy as np
import time
import pandas as pd
from ..methods.muss import (
    greedy_selection,
    mmr_on_partitions,
    partition_random,
    MMR_DIVERSITY_TYPE_SUM,
)

from dppy.finite_dpps import FiniteDPP


def run_random_selection(df, k, dataset="dataset_name", random_state=42):
    """
    Performs random selection of k items from the dataset and evaluates the selection.

    Parameters:
        df (pandas.DataFrame): Input DataFrame to select from
        k (int): Number of items to select
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"

    Returns:
        pandas.DataFrame: Evaluation results as a single-row DataFrame

    Notes:
        Uses fixed random seed (42) for reproducibility
    """
        
    method = 'Random'
    configs = {'method':method,\
              'dataset':dataset,\
               'k': k               
             }
    
    df_selected = df.sample(n=k, random_state=random_state)
    
    return evaluate(df_selected, configs)



def run_experiment_MMR(df,k,lamb=0.5,topk=None ,dataset="dataset_name"):
    """
    Runs Maximal Marginal Relevance (MMR) experiment on the given dataset.

    Parameters:
        df (pandas.DataFrame): Input DataFrame
        k (int): Number of items to select
        lamb (float, optional): Trade-off parameter between relevance and diversity. 
                                  Defaults to 0.5
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"

    Returns:
        pandas.DataFrame: Evaluation results as a single-row DataFrame

    Notes:
        - Processes features and quality scores from input DataFrame
        - Measures and records execution time
        - Returns evaluation metrics for the selected subset
    """

    
    method = 'MMR'
    configs = {'method':method,\
              'dataset':dataset,\
               'k': k,\
               'topk':topk ,\
               'lamb': lamb
             }
    eval_output={}
    
    if topk is not None:
        df.reset_index(inplace=True,drop=True)
        df = df.nlargest(topk, QUALITY_SCORE)
        
    x,q =  process_feature_quality(df)
 
    start_time = time.time()
    selected_idx = greedy_selection(x, q, k=k, weight_q=lamb)

    df_selected = df[df.index.isin(selected_idx)]

    eval_output["time"] = np.round(time.time() - start_time, 2)
    configs.update(eval_output)

    return evaluate(df_selected, configs)



def run_experiment_clustering(df, k, dataset="dataset_name", random_state=321):
    """
    Runs clustering-based selection experiment using FAISS k-means.

    Parameters:
        df (pandas.DataFrame): Input DataFrame
        k (int): Number of clusters and items to select
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"

    Returns:
        pandas.DataFrame: Evaluation results as a single-row DataFrame

    Notes:
        - Uses FAISS k-means with 50 iterations and 2 redos
        - Selects the highest scoring point from each cluster
        - Measures and records execution time
    """
    
    method = 'Clustering'
    configs = {'method':method,'dataset':dataset,'k': k}
    eval_output={}

    x,q =  process_feature_quality(df)
    
    start_time = time.time()

    x = np.ascontiguousarray(x, dtype=np.float32)
    kmeans = faiss.Kmeans(x.shape[1], k, niter=50, verbose=False, nredo=2, seed=random_state)
    kmeans.train(x)

    # get the label assignment
    distance_to_cluster, cluster_assignment = kmeans.index.search(x, 1)
    
    score = q

    selected_point_list = []

    for c in range(k): # loop over clusters

        # find data point index assigned to cluster c
        idx_in_cluster_c = np.where(cluster_assignment == c)[0] 

        # pick the data point which has the highest score
        highest_score_idx_in_c = np.argmax( score[idx_in_cluster_c] )

        # append it to the result
        actual_index = idx_in_cluster_c[highest_score_idx_in_c]
        selected_point_list.append(df.iloc[actual_index, :] )

    df_selected = pd.DataFrame(selected_point_list).reset_index(drop=True)
    
    eval_output={}
    eval_output["time"] = np.round( time.time() - start_time,2)
    configs.update(eval_output)

    return evaluate(df_selected, configs)


def run_experiment_DPP(df, k, dataset="dataset_name", random_state=321):
    """
    Runs Determinantal Point Process (DPP) experiment for diverse subset selection.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing features and quality scores
        k (int): Size of the subset to select
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"

    Returns:
        pandas.DataFrame: Evaluation results as a single-row DataFrame containing:
            - Method configuration parameters
            - Execution time
            - Evaluation metrics for the selected subset

    Notes:
        - Uses likelihood-based DPP for subset selection
        - Constructs L-ensemble matrix as: L = diag(q) * K * diag(q)
          where K = XX^T is the similarity kernel
        - Performs exact k-DPP sampling
        - Measures and records execution time

    Example:
        >>> results = run_experiment_DPP(data_df, k=10, dataset="movies")
        >>> print(results['time'])  # prints execution time
    """
    
    method = 'DPP'
    configs = {'method':method,'dataset':dataset,'k':k }
    eval_output={}

    x,q =  process_feature_quality(df)

    start_time = time.time()

    L = np.diag(q) * x.dot(x.T) * np.diag(q)

    np.random.seed(random_state)
    DPP = FiniteDPP('likelihood', **{'L': L})
    DPP.flush_samples()
    DPP.sample_exact_k_dpp(size=k)

    # this is the list selected indices
    selected_idx = DPP.list_of_samples[0]
    df_selected = df[df.index.isin(selected_idx)]

    eval_output={}
    eval_output["time"] = np.round(time.time() - start_time,2)
    configs.update(eval_output)

    return evaluate(df_selected, configs)
    
    
    
def run_DGDS(df, k=500, k_within=None, n_partitions=200, n_jobs=30, lamb=0.5,
             div_type=MMR_DIVERSITY_TYPE_SUM, random_state=321, dataset="dataset_name"):
    """
    Runs the Distributed Greedy Diversity Selection (DGDS) algorithm.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing features and quality scores
        k (int, optional): Number of items to select. Defaults to 500
        n_partitions (int, optional): Number of random partitions to create. 
                                    Defaults to 200
        lamb (float, optional): Trade-off parameter between relevance and 
                                  diversity. Defaults to 0.5
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"

    Returns:
        pandas.DataFrame: Evaluation results containing:
            - Method configuration parameters
            - Clustering time
            - Maximum partition size
            - Within-partition processing time
            - Union operation time
            - Total execution time

    Notes:
        - Uses random partitioning instead of clustering
        - Applies MMR selection within each partition
        - Measures and records multiple timing metrics
    """
    
    method = 'DGDS'
    configs = {'method':method,\
              'k':k,\
              'l':n_partitions,\
             'lamb':lamb,\
              'dataset':dataset,\
               'n_jobs':n_jobs,\
              'div_type':div_type
             }
    eval_output={}
    
    x,q =  process_feature_quality(df)

    start_time = time.time()
    partition_results = partition_random(x, n_partitions=n_partitions, random_state=random_state)

    eval_output["clustering time"] = np.round( time.time() - start_time,2)

    start_time = time.time()
    
    # take all partitions
    selected_partition_idx =  [partition_results["partitions"][i].astype(int) for i in range(n_partitions)]
        
    
    # perform MMR on partitions
    top_q = set(np.argpartition(q, -k)[-k:].tolist())
    selected_idx, diagnostics = mmr_on_partitions(x, q, k=k, top_q=top_q, lambda_param=lamb,
                                                  n_jobs=n_jobs, k_within=k_within,
                                                  partitions=selected_partition_idx)

    eval_output["time_within"] =  np.round( diagnostics['time_within'],2)
    eval_output["time_union"] =  np.round(diagnostics['time_union'],2)

    df_selected = df[df.index.isin(selected_idx)]
    
    eval_output["time"] = np.round( time.time() - start_time,2)
    configs.update(eval_output)
    
    print("configs",configs)

    return evaluate(df_selected, configs)