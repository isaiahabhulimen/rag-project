import json
import os
from datetime import datetime
from llm_client import client
from config import llm_name
from generator import generate_answer
from retriever import retrieve_chunks
from reranker import rerank_chunks


def answers_match(generated, expected):
    prompt = f"""You are an evaluator. Determine if the Generated Answer matches the Expected Answer in meaning.
Respond with only "YES" or "NO".

Expected Answer: {expected}
Generated Answer: {generated}
"""

    response = client.chat.completions.create(
        model=llm_name, messages=[{"role": "user", "content": prompt}], temperature=0
    )

    return response.choices[0].message.content.strip().upper() == "YES"


def generate_report(eval_results, output_folder="reports"):
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(output_folder, f"eval_report_{timestamp}.json")

    report_data = {
        "timestamp": timestamp,
        "summary": {
            "total": eval_results["total"],
            "passed": eval_results["passed"],
            "failed": eval_results["failed"],
            "accuracy": eval_results["accuracy"],
        },
        "results": eval_results["results"],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"\n{'='*40}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*40}")
    print(f"Total Questions: {eval_results['total']}")
    print(f"Passed: {eval_results['passed']}")
    print(f"Failed: {eval_results['failed']}")
    print(f"Accuracy: {eval_results['accuracy']:.2f}%")
    print(f"\nReport saved to: {report_path}")

    return report_path


def evaluate_rag(
    benchmark, collection, ids, documents, metadatas, model, cross_encoder
):
    total = len(benchmark)
    passed = 0
    failed = 0
    results = []

    for record in benchmark:
        question = record["question"]
        expected_answer = record["expected_answer"]

        combined_chunks, ranked_chunks = retrieve_chunks(
            question, "yes", None, collection, ids, documents, metadatas, model
        )

        combined_chunks = rerank_chunks(
            question, combined_chunks, ranked_chunks, cross_encoder
        )
        context = "\n\n".join(combined_chunks)
        generated_answer = generate_answer(question, context)

        is_match = answers_match(generated_answer, expected_answer)

        if is_match:
            passed += 1
        else:
            failed += 1

        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "passed": is_match,
            }
        )

    accuracy = (passed / total) * 100 if total > 0 else 0

    results_dict = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "results": results,
    }

    generate_report(results_dict)

    return results_dict
