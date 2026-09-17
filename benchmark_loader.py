import json
import os

from config import benchmark_folder


def load_benchmark(file_name):

    benchmark_path = os.path.join(benchmark_folder, file_name)

    with open(benchmark_path, "r", encoding="utf-8") as file:
        benchmark = json.load(file)

    for record in benchmark:

        if "question" not in record:
            raise ValueError("Invalid benchmark record: missing 'question'.")

        if "expected_answer" not in record:
            raise ValueError("Invalid benchmark record: missing 'expected_answer'.")
    return benchmark
