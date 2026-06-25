import argparse
import importlib
import json
import logging
import time

import numpy as np
import pandas as pd
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from typing import Callable, Dict, Optional, Tuple

from muss.src.methods.muss import muss

from muss.src.data_utils import make_data_key
from muss.src.data_utils.embeddings import compute_embeddings, create_index
from muss.src.data_utils.read_data import read_data
from muss.src.llm.bedrock_language_model import BedrockLanguageModel
from muss.src.methods.objectives import global_objective

OUTPUTS_PATH = (Path(__file__).parent / "outputs").resolve()
CACHE_FILE = OUTPUTS_PATH / "_cache.json"

TOP_K = 3
GLOBAL_LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", format="%(name)s: %(message)s", level=logging.DEBUG)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="config file with list of datasets and a list of methods to run"
    )
    parser.add_argument(
        "-d",
        "--data_size",
        type=int,
        default=-1,
        help="optional: restrict dataset size to this number")
    parser.add_argument(
        "-n",
        "--num_questions",
        type=int,
        default=-1,
        help="optional: restrict number of test questions to this number")

    return parser.parse_args()


def _muss_retrieve(
        data_info: Dict,
        query: str,
        k: int,
        m: int = 10,
        lambda_clusters: float = 0.9,
        lambda_points: float = 0.9,
        k_within: Optional[int] = None,
        add_max_q_to_union: bool = True,
) -> Tuple:
    """RAG retrieval using MUSS with pre-computed partitions."""
    x = data_info["pre_proc"]["x"]
    partition_results = data_info["pre_proc"]["partition_results"]
    corpus = data_info["corpus"]
    emb_model = data_info["emb_model"]

    if k > x.shape[0]:
        raise ValueError(f"cannot select more items ({k}) than available: {x.shape[0]}")

    x_query = preprocessing.normalize(np.expand_dims(np.array(emb_model.embed_query(query)), axis=0))
    distances = pairwise_distances(x, x_query, metric="euclidean")[:, 0]
    assert np.max(distances) <= 2, f"embeddings not unit-normalised, max_distance: {np.max(distances)}"
    score = 2 - distances

    t0 = time.time()
    selected_idx, stats = muss(
        x, score, k,
        m=m,
        lamb=lambda_points,
        lamb_c=lambda_clusters,
        k_within=k_within,
        partition_results=partition_results,
        add_max_q_to_union=add_max_q_to_union,
    )
    stats["core_time"] = time.time() - t0
    return [corpus[i] for i in selected_idx], stats


def _get_preprocessor_by_key(method_key: str) -> Optional[Callable]:
    try:
        methods = {
            "muss": ("muss", "pre_process_for_muss")
        }
        module_name, method_name = methods[method_key]
        module = importlib.import_module("muss.src.methods." + module_name)
        return getattr(module, method_name)

    except Exception as e:
        logger.error(e)

    return None


def _get_method_by_key(method_key: str) -> Optional[Callable]:
    if method_key == "muss":
        return _muss_retrieve
    try:
        methods = {
            "random": ("random", "retrieve_random"),
            "dpp": ("dpp", "retrieve_with_dpp"),
        }
        module_name, method_name = methods[method_key]
        module = importlib.import_module("muss.src.methods." + module_name)
        return getattr(module, method_name)

    except Exception as e:
        logger.error(e)

    return None


def _get_cached_answer(llm: BedrockLanguageModel, prompt: str, cache: Dict) -> str:
    if prompt in cache:
        cached_response = cache[prompt]
        if cached_response:
            logger.info("Using cached answer")
            return cached_response

    llm_answer, _ = llm.generate_text(prompt)
    cache[prompt] = llm_answer
    return llm_answer


def _run_configuration(llm: BedrockLanguageModel, data_info: Dict, method_info: Dict) -> Dict:
    method_name = method_info["label"]
    method_key = method_info["method_key"]
    method_params = method_info["parameters"]
    pre_proc_params = method_info.get("pre_proc_params", {})
    logger.info(f"Method {method_name}")

    data_name = data_info["name"]
    questions = data_info["questions"]
    ref_answers = data_info["answers"]

    result = {
        "model_name": llm.model_name,
        "data_name": data_name,
        "corpus_size": len(data_info["corpus"]),
        "num_test_questions": len(data_info["questions"]),
        "method_name": method_name,
        "pre_proc_params": '"' + str(pre_proc_params) + '"',
        "method_params": '"' + str(method_params) + '"'
    }
    method_func = _get_method_by_key(method_key)
    if method_func is None:
        return result

    pre_proc_func = _get_preprocessor_by_key(method_key)
    if pre_proc_func is not None:
        data_info["pre_proc"] = pre_proc_func(data_info, **pre_proc_params)
    else:
        data_info["pre_proc"] = {}

    trace_records = []
    num_total = len(questions)
    num_correct = 0
    num_empty = 0
    objective_totals = [0.0] * len(GLOBAL_LAMBDA_VALUES)

    if CACHE_FILE.is_file():
        cache = json.loads(CACHE_FILE.read_text())
    else:
        cache = {}

    for i, qa_pair in enumerate(zip(questions, ref_answers)):
        logger.info(f"~~~~~ {i + 1} out of {num_total}")
        question, ref_answer = qa_pair[0], qa_pair[1]
        start_time = time.time()
        rag_results, round_stats = method_func(data_info, question, TOP_K, **method_params)
        round_stats["elapsed"] = time.time() - start_time

        objective_values = global_objective(question, rag_results, data_info["emb_model"], GLOBAL_LAMBDA_VALUES)
        for i_lambda in range(len(GLOBAL_LAMBDA_VALUES)):
            objective_totals[i_lambda] += objective_values[i_lambda]

        rag_results.reverse()  # most relevant fact closer to the question
        prompt = (
            "<task>\n"
            + "You will be given a question and additional information to consider.\n"
            + "This information might or might not be relevant to the question.\n"
            + "Your task is to answer the question.\n"
            + "Only use additional information if it's relevant.\n"
            + "</task>\n"
            + "<additional_information>\n"
            + "\n".join(rag_results)
            + "\n"
            + "</additional_information>\n"
            + "<question>\n"
            + question
            + "</question>\n"
            + "<output_format>\n"
            + "In your response, only include the answer itself. No tags, no other words.\n"
            + "</output_format>\n"
        )
        llm_answer = _get_cached_answer(llm, prompt, cache)

        if llm_answer:
            match = llm_answer.strip().lower() == ref_answer.strip().lower()
            num_correct += int(match)
        else:
            match = None
            num_empty += 1

        record = {
            "question": question,
            "ref_answer": ref_answer,
            "rag_results": rag_results,
            "llm_answer": llm_answer,
            "match": match,
            "round_stats": round_stats
        }
        trace_records.append(record)
        logger.info("~" * 30)

    out_file = OUTPUTS_PATH / f"{data_name}~{method_name}~answers.json"
    out_file.write_text(json.dumps(trace_records, indent=2))
    CACHE_FILE.write_text(json.dumps(cache))

    accuracy = num_correct / (num_total - num_empty) if (num_total - num_empty) > 0 else 0.0
    metrics = {"accuracy": accuracy, "num_empty": num_empty}
    for l_param, v_objective in zip(GLOBAL_LAMBDA_VALUES, objective_totals):
        metrics[f"obj_{l_param:0.1f}"] = v_objective
    if "stats" in data_info["pre_proc"]:
        for key, value in data_info["pre_proc"]["stats"].items():
            metrics[key] = value
    partial_sums = {}
    for trace_record in trace_records:
        for key, value in trace_record["round_stats"].items():
            s = partial_sums.get(key, 0) + value
            partial_sums[key] = s
    for key, value in partial_sums.items():
        metrics[key] = value / len(trace_records)
    result.update(metrics)
    return result


def main() -> int:
    args = _parse_arguments()
    setup = json.loads(Path(args.input).read_text())
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in setup["models"]:
        llm = BedrockLanguageModel(model_name=model_name)
        for dataset_name in setup["datasets"]:
            corpus, questions, ref_answers, multiple_refs = read_data(dataset_name)
            if (args.data_size > 0) and (len(corpus) > args.data_size):
                corpus = corpus[:args.data_size]
                logger.info(f"Restricting the corpus size to {args.data_size}")
            if (args.num_questions > 0) and (len(questions) > args.num_questions):
                questions = questions[:args.num_questions]
                ref_answers = ref_answers[:args.num_questions]
                logger.info(f"Restricting number of test questions to {args.num_questions}")

            emb_model = HuggingFaceEmbeddings()
            data_key = make_data_key(dataset_name, emb_model.model_name, corpus)
            text_embeddings = compute_embeddings(corpus, emb_model, data_key)
            vector_db = create_index(text_embeddings, emb_model, data_key)

            data_info = {
                "name": dataset_name,
                "corpus": corpus,
                "questions":  questions,
                "answers": ref_answers,
                "multiple_refs": multiple_refs,
                "data_key": data_key,
                "emb_model": emb_model,
                "text_embeddings": text_embeddings,
                "vector_db": vector_db,
            }
            for method_info in setup["methods"]:
                result = _run_configuration(llm, data_info, method_info)
                logger.info(result)
                results.append(result)

                # make results available for review after each iteration
                out_file = f"results~{Path(args.input).stem}.csv"
                pd.DataFrame(results).to_csv(OUTPUTS_PATH / out_file, index=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
