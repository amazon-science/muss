import numpy as np

from dppy.finite_dpps import FiniteDPP
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from typing import Dict


def retrieve_with_dpp(data_info: Dict, query: str, k:int):
    # corpus embeddings as a single data matrix
    text_embeddings = data_info["text_embeddings"]
    x_list = [item[1] for item in text_embeddings]
    x = np.array(x_list)  # 2D array (n items, d features)
    x = preprocessing.normalize(x)

    # obtain the quality score = negative Euclidean distance to query
    emb_model = data_info["emb_model"]
    x_query = np.array(emb_model.embed_query(query))
    x_query = np.expand_dims(x_query, axis=0)  # 2D array (1 item, d features)
    x_query = preprocessing.normalize(x_query)

    d = pairwise_distances(x, x_query, metric="euclidean")  # 2D array (n items, 1)
    distances = d[:,0]  # 1D array (n items)
    max_distance = np.max(distances)
    # maximum Euclidean distance between two unit-normalized vectors is 2
    assert max_distance <= 2, f"not properly scaled, max_distance: {max_distance}"
    q = 2 - distances

    # seed depends on query, so that results are random, but reproducible for a given query
    seed = 0
    for i, c in enumerate(query[:3]):
        seed += 321**i * ord(c)

    matrix_l = np.diag(q) * x.dot(x.T) * np.diag(q)
    dpp = FiniteDPP("likelihood", **{"L": matrix_l})
    dpp.flush_samples()
    dpp.sample_exact_k_dpp(size=k, random_state=seed)
    selected_indices = dpp.list_of_samples[0]

    corpus = data_info["corpus"]
    return [corpus[i] for i in selected_indices], {}