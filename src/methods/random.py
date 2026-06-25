import random

from typing import Dict, Tuple


def retrieve_random(data_info: Dict, query: str, k:int, num_results: int=-1) -> Tuple:
    if num_results == -1:
        num_results = k
    else:
        assert num_results < k, f"top k = {k}, num random results = {num_results}"

    corpus_items = data_info["corpus"]

    # seed depends on query, so that results are random, but reproducible for a given query
    seed = 0
    for i, c in enumerate(query[:3]):
        seed += 321**i * ord(c)
    random.seed(seed)

    return random.sample(corpus_items, num_results), {}
