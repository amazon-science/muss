import pickle

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import List


VECTOR_DB_PATH = (Path(__file__).parent / ".." / ".." / "outputs" / "vector_db").resolve()


def compute_embeddings(corpus_items: List, emb_model: HuggingFaceEmbeddings, data_key:str) -> List:
    emb_file = VECTOR_DB_PATH / f"emb~{data_key}.pkl"
    if emb_file.is_file():
        with open(emb_file, "rb") as f:
            return pickle.load(f)

    embeddings = emb_model.embed_documents(corpus_items)
    text_embeddings = list(zip(corpus_items, embeddings))

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    with open(emb_file, "wb") as f:
        pickle.dump(text_embeddings, f)

    return text_embeddings


def create_index(text_embeddings: List, emb_model: HuggingFaceEmbeddings, data_key:str) -> FAISS:
    index_path = VECTOR_DB_PATH / f"{data_key}"
    if index_path.is_dir():
        return FAISS.load_local(str(index_path), emb_model, allow_dangerous_deserialization=True)

    doc_ids = [{"id": i} for i in range(len(text_embeddings))]
    vector_db = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=emb_model, metadatas=doc_ids)
    vector_db.save_local(str(index_path))
    return vector_db
