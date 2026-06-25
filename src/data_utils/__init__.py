import hashlib

from typing import List


def get_data_hash(records: List) -> str:
    m = hashlib.sha256()
    hashed_records = records[:30]
    hashed_records += records[-30:]
    for record in hashed_records:
        m.update(record.encode())
    data_hash = m.hexdigest()[:8]
    return str(data_hash)


def make_data_key(data_name: str, emb_model_name: str, corpus_items: List) -> str:
    model_name = emb_model_name.replace("/", "_")
    data_hash = get_data_hash(corpus_items)
    return f"{data_name}~{len(corpus_items)}~{data_hash}~{model_name}"
