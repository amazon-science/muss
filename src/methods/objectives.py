import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from typing import List


def global_objective(query: str, rag_results: List, emb_model: HuggingFaceEmbeddings, lambdas: List) -> List:
    x_list = [emb_model.embed_query(t) for t in rag_results]
    x = np.array(x_list)  # 2D array (n items, d features)
    x = preprocessing.normalize(x)

    d = pairwise_distances(x, metric="euclidean")  # (n items, n items)
    diversity = np.mean(d)

    # adjust for zeros on the diagonal
    n = len(x_list)
    diversity = diversity * (n * n) / (n * n - n)

    q_emb = emb_model.embed_query(query)
    q = np.array(q_emb).reshape(1, -1)
    q = preprocessing.normalize(q)
    r = 2 - pairwise_distances(q, x, metric="euclidean")
    relevance = np.mean(r)

    values = []
    for l_param in lambdas:
        assert (l_param >= 0.0) and (l_param <= 1.0), f"l_param = {l_param}"
        values.append(l_param * relevance + (1 - l_param) * diversity)

    return values
