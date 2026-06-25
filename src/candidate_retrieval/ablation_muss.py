from .helpers import process_feature_quality, evaluate
import time
import numpy as np
from ..methods.muss import partition_clustering_by_faiss, select_partitions_mmr, partition_random, mmr_on_partitions

def run_MUSS_random_cluster_selection(df, k=500, k_within=None, m=100, n_partitions=200,
                                      lamb=0.5, lamb_c=0.5, n_jobs=25,
                                      random_state=321, dataset="dataset_name"):
    """
    Runs MUSS with random cluster selection instead of MMR-based cluster selection.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing features and quality scores
        k (int, optional): Size of final subset to select. Defaults to 500
        m (int, optional): Number of clusters to randomly select. Defaults to 100
        n_partitions (int, optional): Total number of clusters to create. 
                                    Defaults to 200
        lamb (float, optional): Individual MMR trade-off parameter. 
                                  Defaults to 0.5
        lamb_c (float, optional): Not used in this variant since selection 
                                      is random. Defaults to 0.5
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"

    Returns:
        pandas.DataFrame: Evaluation results containing:
            - Method configuration parameters
            - Clustering time
            - Maximum cluster size
            - Within-cluster processing time
            - Union operation time
            - Total execution time

    Notes:
        Algorithm steps:
        1. Performs clustering using FAISS
        2. Randomly selects m clusters (instead of using MMR)
        3. Applies MMR within selected clusters
        4. Records various timing metrics
    """
    
    method = 'MUSS Rand Cluster'
    configs = {'method':method,\
              'm':m,'k':k,\
              'l':n_partitions,\
             'lamb':lamb,\
              'lamb_c':lamb_c,
              'dataset':dataset
             }
        
    eval_output={}
    x,q =  process_feature_quality(df)

    start_time = time.time()
    cluster_out = partition_clustering_by_faiss(x, n_partitions=n_partitions, random_state=random_state)
    eval_output["clustering time"] = np.round(time.time() - start_time, 2)

    start_time = time.time()
    rng = np.random.default_rng(random_state)
    n_avail = len(cluster_out["partitions"])
    selected_partition = rng.choice(n_avail, min(m, n_avail), replace=False)
    selected_partition_idx = [cluster_out["partitions"][i] for i in selected_partition]

    top_q = set(np.argpartition(q, -k)[-k:].tolist())
    selected_idx, diagnostics = mmr_on_partitions(x, q, k=k, top_q=top_q,
                                                  lambda_param=lamb, n_jobs=n_jobs,
                                                  k_within=k_within,
                                                  partitions=selected_partition_idx)

    eval_output["time_within"] = np.round(diagnostics['time_within'], 2)
    eval_output["time_union"] = np.round(diagnostics['time_union'], 2)

    df_selected = df[df.index.isin(selected_idx)]

    eval_output["time"] = np.round(time.time() - start_time, 2)
    configs.update(eval_output)

    return evaluate(df_selected, configs), df_selected



def run_MUSS_rand_partition(df, k=500, k_within=None, m=100, n_partitions=200, lamb=0.5,
                           lamb_c=0.5, n_jobs=25, random_state=321,
                           dataset="dataset_name", flagReturnDf=False):
    """
    Runs MUSS (Multi-Scale Subset Selection) with random partitioning instead of clustering.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing features and quality scores
        k (int, optional): Size of final subset to select. Defaults to 500
        m (int, optional): Number of partitions to select. Defaults to 100
        n_partitions (int, optional): Total number of random partitions to create. 
                                    Defaults to 200
        lamb (float, optional): Individual MMR trade-off parameter. 
                                  Defaults to 0.5
        lamb_c (float, optional): Partition selection trade-off parameter. 
                                      Defaults to 0.5
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"
        flagReturnDf (bool, optional): If True, returns selected indices; 
                                     if False, returns evaluation results. 
                                     Defaults to False

    Returns:
        Union[list, pandas.DataFrame]: 
            - If flagReturnDf is True: List of selected indices
            - If flagReturnDf is False: DataFrame with evaluation metrics

    Notes:
        Algorithm steps:
        1. Creates random partitions instead of clustering
        2. Selects m partitions using MMR
        3. Applies MMR within selected partitions
        4. Records timing metrics for each step
    """
    
    method = 'MUSS Rand Partition'
    configs = {'method':method,\
              'm':m,'k':k,\
              'l':n_partitions,\
             'lamb':lamb,\
              'lamb_c':lamb_c,
              'dataset':dataset
             }
        
    eval_output={}
    
    x,q =  process_feature_quality(df)

    start_time = time.time()
    partition_results = partition_random(x, n_partitions=n_partitions, random_state=random_state)

    eval_output["clustering time"] = np.round(time.time() - start_time, 2)

    start_time = time.time()
    selected_partition = select_partitions_mmr(q, partition_results=partition_results, m=m, lambda_part=lamb_c)
    selected_partition_idx = [partition_results["partitions"][i] for i in selected_partition]

    top_q = set(np.argpartition(q, -k)[-k:].tolist())
    selected_idx, diagnostics = mmr_on_partitions(x, q, k=k, top_q=top_q,
                                                  lambda_param=lamb, n_jobs=n_jobs,
                                                  k_within=k_within,
                                                  partitions=selected_partition_idx)

    eval_output["time_within"] = np.round(diagnostics['time_within'], 2)
    eval_output["time_union"] = np.round(diagnostics['time_union'], 2)

    df_selected = df[df.index.isin(selected_idx)]

    eval_output["time"] = np.round(time.time() - start_time, 2)
    configs.update(eval_output)

    print("configs", configs)

    if flagReturnDf:
        return selected_idx
    return evaluate(df_selected, configs), df_selected