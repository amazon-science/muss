import time
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union
from scipy.stats import entropy

from ..methods.muss import muss, MMR_DIVERSITY_TYPE_SUM, MMR_DIVERSITY_TYPE_MIN

QUALITY_SCORE = "quality_score"
EMBEDDING_COLUMN = 'emb'
LABEL_COLUMN = "label"
ASPECT_COLUMN = 'subcategory_code'
MAX_RELEVANT_SCORE = 0.01

def get_obj_score(df):
    """
    Calculate an objective score for each row in the dataframe based on quality and pairwise distance metrics.
    
    The objective score is a weighted combination of quality and pairwise distance:
    - If 'lamb' values exist: score = quality * lamb + (1-lamb) * pw_dist
    - If 'lamb' values are missing: score = 0.5 * quality + 0.5 * pw_dist
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing columns:
        - 'quality': Quality scores
        - 'pw_dist': Pairwise distance scores
        - 'lamb': Lambda weight values (optional)
        
    Returns
    -------
    pandas.DataFrame
        Input dataframe with an additional 'obj' column containing the calculated
        objective scores rounded to 6 decimal places.
    
    Notes
    -----
    The function modifies the input dataframe by adding/updating the 'obj' column.
    """
    if not df["lamb"].isna().any():
        df['obj'] = df['quality']*df['lamb'] + (1-df['lamb'])*df['pw_dist']
    else:
        df['obj'] = df['quality']*0.5 + 0.5*df['pw_dist']
    
    df['obj'] = np.round(df['obj'],6)
    return df
    
def process_feature_quality(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficiently process feature vectors and quality scores from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing embeddings and quality scores
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature vectors array and quality scores array
        
    Note:
        Uses vectorized operations and memory-efficient numpy arrays
    """
    # Convert embeddings to numpy array directly using stack for better performance
    try:
        # Faster method if embeddings are already in correct format
        x = np.stack(df[EMBEDDING_COLUMN].values)
    except ValueError:
        # Fallback method if embeddings need conversion
        x = np.array([np.array(v, dtype=np.float32) for v in df[EMBEDDING_COLUMN].values])
    
    # Convert quality scores to float32 for memory efficiency
    q = df[QUALITY_SCORE].values.astype(np.float32)
    
    return x, q



def clustering_evaluation(item_feats: Union[np.ndarray, pd.DataFrame]) -> Tuple[float, float, float]:
    """
    Evaluate clustering metrics for a set of feature vectors.
    
    This function computes three different distance-based metrics:
    1. Average pairwise distance between all points
    2. Average nearest neighbor distance
    3. Average minimum distance
    
    Args:
        item_feats: Feature vectors with shape (N, d) where:
            - N is the number of items
            - d is the dimension of feature vectors
            Can be either numpy array or pandas DataFrame.
    
    Returns:
        Tuple containing three rounded metrics (to 4 decimal places):
            - Average pairwise distance
            - Average nearest neighbor distance
            - Average minimum distance
    
    Example:
        >>> features = np.random.rand(100, 10)
        >>> avg_dist, nn_dist, min_dist = clustering_evaluation(features)
    """
    s_dist = average_distance(item_feats)
    s_nn_dist = average_nearest_neighbor_distance(item_feats)
    s_min_dist = average_min_distance(item_feats)
    
    return np.round(s_dist, 4), np.round(s_nn_dist, 4), np.round(s_min_dist, 4)

def average_distance(item_feats: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate average pairwise Euclidean distance between all points.
    
    Args:
        item_feats: Feature vectors with shape (N, d) where:
            - N is the number of items
            - d is the dimension of feature vectors
            Can be either numpy array or pandas DataFrame.
    
    Returns:
        float: Average pairwise distance normalized by N^2
    
    Notes:
        - If input is DataFrame, converts to numpy array
        - Uses vectorized operations for efficient computation
        - Distance calculation: sqrt(sum((x_i - x_j)^2))
    """
    if isinstance(item_feats, pd.DataFrame):
        item_feats = item_feats.values
    
    N = item_feats.shape[0]
    
    # Compute pairwise Euclidean distances efficiently
    distances = np.linalg.norm(item_feats[:, np.newaxis] - item_feats, axis=2)
    
    return np.sum(distances) / (N * N)

def average_nearest_neighbor_distance(item_feats: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate average distance to nearest neighbor for each point.
    
    Args:
        item_feats: Feature vectors with shape (N, d) where:
            - N is the number of items
            - d is the dimension of feature vectors
            Can be either numpy array or pandas DataFrame.
    
    Returns:
        float: Average nearest neighbor distance
    
    Notes:
        - Excludes self-distances by setting diagonal to infinity
        - For each point, finds the closest other point
        - Returns average of these minimum distances
        - Handles both numpy arrays and pandas DataFrames
    """
    if isinstance(item_feats, pd.DataFrame):
        item_feats = item_feats.values
    
    # Calculate pairwise Euclidean distances
    all_distances = np.linalg.norm(item_feats[:, np.newaxis] - item_feats, axis=2)
    
    # Exclude self-distances
    np.fill_diagonal(all_distances, np.inf)
    
    # Find nearest neighbors
    nearest_indices = np.argmin(all_distances, axis=1)
    
    # Calculate average distance to nearest neighbors
    total_distances = np.sum(all_distances[np.arange(len(item_feats)), nearest_indices])
    
    return total_distances / len(item_feats)

def average_min_distance(item_feats: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate average of minimum distances for each point.
    
    Args:
        item_feats: Feature vectors with shape (N, d) where:
            - N is the number of items
            - d is the dimension of feature vectors
            Can be either numpy array or pandas DataFrame.
    
    Returns:
        float: Average of minimum distances
    
    Notes:
        - Similar to nearest neighbor distance but uses mean instead of sum
        - Excludes self-distances by setting diagonal to infinity
        - More efficient implementation using numpy's min and mean
    """
    if isinstance(item_feats, pd.DataFrame):
        item_feats = item_feats.values
    
    # Calculate pairwise Euclidean distances
    all_distances = np.linalg.norm(item_feats[:, np.newaxis] - item_feats, axis=2)
    np.fill_diagonal(all_distances, np.inf)
    
    # Find minimum distance for each point and average
    min_per_row = np.min(all_distances, axis=1)
    
    return np.mean(min_per_row)


def compute_log2_lookup(max_size: int) -> np.ndarray:
    """Cached computation of log2(n+1) values."""
    return np.log2(np.arange(1, max_size + 2))

def fast_alpha_ndcg_calc(targets: np.ndarray, 
                        positions: np.ndarray,
                        aspects: np.ndarray,
                        log2_vals: np.ndarray,
                        alpha: float,
                        k: int) -> float:
    """Optimized α-NDCG calculation using Numba."""
    n_items = min(len(targets), k)
    alpha_dcg = 0.0
    
    unique_aspects = np.unique(aspects[:n_items])
    
    for asp in unique_aspects:
        aspect_mask = aspects[:n_items] == asp
        if not np.any(aspect_mask):
            continue
            
        prev_seen = np.arange(0, np.sum(aspect_mask))
        relevance = targets[aspect_mask] * (1 - alpha) ** prev_seen
        pos_indices = positions[aspect_mask]
        alpha_dcg += np.sum(relevance / log2_vals[pos_indices])
    
    return alpha_dcg

def alpha_ndcg(y_true: np.ndarray,
               y_pred: np.ndarray,
               aspect: np.ndarray,
               max_relevant_score: float = MAX_RELEVANT_SCORE,
               alpha: float = 0.5,
               k: int = 5) -> float:
    """
    Optimized implementation of α-NDCG calculation.
    
    Args:
        y_true: Target values array
        y_pred: Predicted positions array
        aspect: Aspect categories array
        max_relevant_score: Maximum relevance score
        alpha: Diversity parameter
        k: Number of top items to consider
    
    Returns:
        float: α-NDCG score
    """
    y_pred = np.asarray(y_pred)
    if y_pred.min() == 0:
        y_pred = y_pred + 1
        
    # Pre-compute log2 values
    log2_vals = compute_log2_lookup(y_pred.max())
    
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float32)
    aspect = np.asarray(aspect)
    
    # Calculate α-DCG
    alpha_dcg = fast_alpha_ndcg_calc(
        y_true, y_pred, aspect, log2_vals, alpha, k
    )
    
    # Calculate ideal DCG
    n_aspects = len(np.unique(aspect[:k]))
    n_items = min(len(y_true), k)
    
    if k >= n_aspects:
        ideal_ndcg = n_items * max_relevant_score * (1 - alpha) / np.log2(2)
    else:
        ave_prev_seen = int(np.ceil(n_items / n_aspects))
        positions = np.arange(1, ave_prev_seen)
        ideal_ndcg = np.sum(
            n_aspects * max_relevant_score * (1-alpha)**positions /
            log2_vals[positions]
        )
    
    return np.round(alpha_dcg / ideal_ndcg, 5)

def evaluation_selected_items(selected_df: pd.DataFrame,
                            configs: Dict[str, Union[int, float]]) -> pd.DataFrame:
    """
    Optimized evaluation of selected items.
    
    Args:
        selected_df: DataFrame with selected items
        configs: Configuration dictionary
    
    Returns:
        DataFrame with evaluation metrics
    """
    k = configs["k"]
    
    # Efficient sorting and slicing
    selected_df = (selected_df
                  .nlargest(k, QUALITY_SCORE)
                  .reset_index(drop=True))
    
    # Compute all metrics in parallel
    feat = np.asarray(selected_df[EMBEDDING_COLUMN].tolist(), dtype=np.float32)
    
    # Parallel computation of metrics
    results = {
        'quality': np.round(selected_df[QUALITY_SCORE].mean(), 5),
        'min_qual': np.round(selected_df.quality_score.min(),5),
        'a_ndcg': alpha_ndcg(
            selected_df.ctr.values,
            selected_df.index.values,
            selected_df.subcategory_code.values,
            k=k
        ),
        'pw_dist': (cluster_eval := clustering_evaluation(feat))[0],
        'min_dist': cluster_eval[2]
    }
    
    # Compute precision metrics efficiently
    for k_value in [100, 300, 500]:
        results[f'P@{k_value}'] = estimate_precision(
            selected_df[LABEL_COLUMN][:k_value].values
        )

    # entropy per categorical column (only if present in the dataframe)
    for col, key in [
        ("subcategory_code", "ent_subcat"),
        ("cat_a",            "ent_cat_a"),
        ("cat_b",            "ent_cat_b"),
        ("cat_c",            "ent_cat_c"),
    ]:
        if col in selected_df:
            results[key] = calculate_entropy(selected_df[col])

    return pd.DataFrame([results])

def estimate_precision(y_true: np.ndarray) -> float:
    """Optimized precision calculation."""
    return np.round(100 * np.mean(y_true), 1)


    
def calculate_entropy(df_items):
    """
    Calculate the Shannon entropy of a categorical feature in the dataframe.
    
    The entropy measures the level of uncertainty or randomness in the data distribution.
    A value of 0 indicates perfect purity (all samples belong to the same category),
    while higher values indicate more diverse distributions.
    
    Parameters
    ----------
    df_items : pandas.Series
        A single column/feature of categorical values with shape [N x 1],
        where N is the number of samples.
    
    Returns
    -------
    float
        The calculated entropy value rounded to 3 decimal places.
        The entropy is computed as: -sum(p_i * log2(p_i))
        where p_i is the probability of each unique value.
    
    Examples
    --------
    >>> df = pd.Series(['A', 'A', 'B', 'C'])
    >>> calculate_entropy(df)
    1.500
    
    Notes
    -----
    - Uses scipy.stats.entropy under the hood
    - For a completely uniform distribution, entropy will be maximum
    - For a single-value distribution, entropy will be 0
    """

    t = np.array(df_items.value_counts())
    return np.round(entropy(t / t.sum()), 3)
    

def evaluate(df_selected, configs):
    """
    Evaluates the selected items and updates the configuration dictionary with evaluation metrics.

    Parameters:
        df_selected (pandas.DataFrame): DataFrame containing the selected items
        configs (dict): Configuration dictionary containing experiment parameters

    Returns:
        pandas.DataFrame: A single-row DataFrame containing all configurations and evaluation metrics
    """
    
    eval_output = evaluation_selected_items(df_selected, configs)
    configs.update(eval_output.iloc[0].to_dict())
    return pd.DataFrame([configs])



def save_result(output_df):
    """
    Saves experiment results to a CSV file after cleaning and processing the data.

    Parameters:
        output_df (pandas.DataFrame): DataFrame containing experiment results with 
                                    at least a 'dataset' column

    Notes:
        - Saves file to 'temp/{dataset_name}.csv'
        - Fills NA values with 0
        - Removes duplicate rows based on specific columns
        - Removes any 'unnamed' columns
        - Performs in-place operations on the DataFrame
    """
        
    path = 'temp/{}.csv'.format(output_df.iloc[0]["dataset"])
    Path("temp").mkdir(exist_ok=True)
    print(path)
    
    output_df = output_df.fillna(0)

    output_df.drop_duplicates(inplace=True)
        
    try:
        output_df.drop_duplicates(subset=['method','dataset','k', 'l','m',\
                          'lamb','lamb_c','div_type','random_state'], keep='last',inplace=True)
    except KeyError:
        pass

    try:
        output_df.drop(output_df.columns[output_df.columns.str.contains(
            'unnamed', case=False)], axis=1, inplace=True)
    except KeyError:
        pass
            

    output_df.to_csv(path)
    



def run_MUSS(df, k=500, k_within=500, m=100, n_partitions=200, lamb=0.5, lamb_c=0.5,
             n_jobs=25, topk=None, dataset="dataset_name",
             div_type=MMR_DIVERSITY_TYPE_SUM, is_clustering_q=False,
             random_state=321):
    """
    Runs the Multi-Scale Subset Selection (MUSS) algorithm.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing features and quality scores
        k (int, optional): Size of final subset to select. Defaults to 500
        m (int, optional): Number of partitions to select. Defaults to 100
        n_partitions (int, optional): Total number of clusters to create. 
                                    Defaults to 200
        lamb (float, optional): Individual MMR trade-off parameter. 
                                  Defaults to 0.5
        lamb_c (float, optional): Cluster selection trade-off parameter. 
                                      Defaults to 0.5
        dataset (str, optional): Name of the dataset. Defaults to "dataset_name"
        diversity_type (str, optional): Type of diversity. Either MMR_DIVERSITY_TYPE_SUM or MMR_DIVERSITY_TYPE_MIN
        

    Returns:
        Union[pandas.DataFrame, pandas.DataFrame]: 
            DataFrame with evaluation results, DataFrame with selected items

    Notes:
        Algorithm steps:
        1. Performs clustering using FAISS
        2. Selects m partitions using MMR
        3. Applies MMR within selected partitions
        4. Records various timing metrics throughout the process
    """
    
    method = 'MUSS'
    configs = {'method':method,\
              'm':m,
               'k':k,
               'k_within':k_within,\
               'topk':topk,\
              'l':n_partitions,\
             'lamb':lamb,\
              'lamb_c':lamb_c,\
              'dataset':dataset,\
              'n_jobs':n_jobs,\
               'div_type':div_type,\
               'is_clustering_q':is_clustering_q,\
               'random_state':random_state
             }
        
    eval_output={}
    
    if topk is not None:
        df.reset_index(inplace=True,drop=True)
        df = df.nlargest(topk, QUALITY_SCORE)

    x,q =  process_feature_quality(df)

    
    if is_clustering_q:
        feat_for_clustering = np.hstack( (x, np.reshape(q,(-1,1))) )
    else:
        feat_for_clustering = x
        
    start_time = time.time()
    
    selected_idx, diagnostics = muss(feat_for_clustering, q, k, k_within, m, n_partitions,
                                     lamb, lamb_c, n_jobs, diversity_type=div_type,
                                     random_state=random_state)
        
    eval_output["time_within"] =  np.round( diagnostics['time_within'],2)
    eval_output["time_union"] =  np.round(diagnostics['time_union'],2)
    
    df_selected = df[df.index.isin(selected_idx)]
    
    eval_output["clustering time"] = diagnostics["clustering_time"]
    eval_output["time"] = np.round( time.time() - start_time,2)
    configs.update(eval_output)

    print("configs",configs)
    
    return evaluate(df_selected, configs), df_selected
