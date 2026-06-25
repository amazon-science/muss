import json
import sys

from pathlib import Path
from typing import List, Set


OUTPUTS_PATH = (Path(__file__).parent / ".." / "outputs").resolve()

METHOD_NAME_MAP = {
    "packing_random": "random",
    "packing_k_means": "clustering",
}

BATCH_SIZE = 100


def _make_configs_for_lambdas(methods: List, labels: Set) -> None:
    n_clusters, m = 100, 10
    for l_c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for l_p in [0.5, 0.7]:
            for algo in METHOD_NAME_MAP.keys():
                setting = {
                    "label": f"MUSS ({METHOD_NAME_MAP[algo]})~{n_clusters:03d}~{m}~{l_c:0.1f}~{l_p:0.1f}",
                    "method_key": "muss",
                    "pre_proc_params": {
                        "n_clusters": n_clusters,
                        "algorithm": algo
                    },
                    "parameters": {
                        "m": m,
                        "lambda_clusters": l_c,
                        "lambda_points": l_p
                    }
                }
                if setting["label"] not in labels:
                    methods.append(setting)
                    labels.add(setting["label"])


def _make_configs_for_m_n(methods: List, labels: Set) -> None:
    l_c, l_p = 0.7, 0.7
    for n_clusters in [50, 100, 200]:
        for m in [10, 20]:
            for algo in METHOD_NAME_MAP.keys():
                setting = {
                    "label": f"MUSS ({METHOD_NAME_MAP[algo]})~{n_clusters:03d}~{m}~{l_c:0.1f}~{l_p:0.1f}",
                    "method_key": "muss",
                    "pre_proc_params": {
                        "n_clusters": n_clusters,
                        "algorithm": algo
                    },
                    "parameters": {
                        "m": m,
                        "lambda_clusters": l_c,
                        "lambda_points": l_p
                    }
                }
                if setting["label"] not in labels:
                    methods.append(setting)
                    labels.add(setting["label"])


def main() -> int:
    methods = []
    labels = set()
    _make_configs_for_lambdas(methods, labels)
    _make_configs_for_m_n(methods, labels)

    print(len(methods))
    for i in range(0, len(methods), BATCH_SIZE):
        batch = methods[i:i+BATCH_SIZE]
        print(len(batch))
        settings = {
            "models": ["us.anthropic.claude-3-5-haiku-20241022-v1:0"],
            "datasets": ["StackExchange", "DevOps"],
            "methods": batch
        }
        OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUTS_PATH / f"ours-{i:03d}.json"
        out_file.write_text(json.dumps(settings, indent=2))

    return 0


if __name__ == '__main__':
    sys.exit(main())
