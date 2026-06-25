import json
import logging
import nltk

from pathlib import Path
from typing import List, Tuple


logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", format="%(name)s: %(message)s", level=logging.DEBUG)

DATASETS_PATH = (Path(__file__).parent / ".." / ".." / "data").resolve()


def read_data(dataset_name: str, show_preview: bool = True) -> Tuple:
    data_settings = {
        "StackExchange": {"reader": read_stack_exchange_exam, "multiple_refs": False},
        "DevOps": {"reader": read_devops_exam, "multiple_refs": False},
    }
    corpus, questions, ref_answers = data_settings[dataset_name]["reader"]()
    multiple_refs = data_settings[dataset_name]["multiple_refs"]

    if show_preview:
        _preview_dataset(dataset_name, corpus, questions, ref_answers)

    return corpus, questions, ref_answers, multiple_refs


def _preview_dataset(dataset_name: str, corpus: List, questions: List, ref_answers: List) -> None:
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Corpus size: {len(corpus)}")
    logger.info(f"Num questions: {len(questions)}")
    logger.info(f"Corpus preview:")
    for fact in corpus[:3]:
        logger.info(f"\t{fact}")
    logger.info(f"Questions preview:")
    for q, a in zip(questions[:3], ref_answers[:3]):
        logger.info("\t" + "~" * 50 + "\n")
        logger.info(f"\t{q}")
        logger.info(f"\t{a}")


def read_devops_exam() -> Tuple:
    # noinspection SpellCheckingInspection
    exam_path = DATASETS_PATH / "DevOps" / "ExamData" / "html_llamav2_2023091421"
    corpus, questions, ref_answers = read_exam_data(exam_path)
    return corpus, questions, ref_answers


def read_stack_exchange_exam() -> Tuple:
    # noinspection SpellCheckingInspection
    exam_path = DATASETS_PATH / "StackExchange" / "ExamData" / "llamav2_2023091223"
    corpus, questions, ref_answers = read_exam_data(exam_path)
    return corpus, questions, ref_answers


def read_exam_data(exam_path: Path) -> Tuple:
    records = (exam_path / "exam.json").read_text()
    records = json.loads(records)

    # noinspection SpellCheckingInspection
    nltk.download("punkt_tab")

    corpus, questions, ref_answers = [], [], []
    for record in records:
        question = record["question"] + "\n"
        question += "Select one of the following answers:\n"
        question += "\n".join(record["choices"])
        question += "\n"
        questions.append(question)

        sentences = nltk.sent_tokenize(record["documentation"])
        corpus.extend(sentences)

        ref_answers.append(record["correct_answer"][0])

    corpus = [d.strip() for d in corpus if d.strip()]

    return corpus, questions, ref_answers
