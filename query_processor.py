from config import llm_name
from llm_client import chat


def route_question(question):
    prompt = f"""Classify this question as "SIMPLE" or "COMPLEX".
Respond with only one word.

SIMPLE = a single clear question asking about one fact or concept.
COMPLEX = contains multiple parts, comparisons, or requires multiple steps to answer.

Question: {question}
Classification:"""

    response = chat(
        model=llm_name, messages=[{"role": "user", "content": prompt}], temperature=0
    )

    result = response.choices[0].message.content.strip().upper()
    return "COMPLEX" if "COMPLEX" in result else "SIMPLE"


def decompose_question(question):
    prompt = f"""Break this complex question into simpler sub-questions that can be answered independently.
Return each sub-question on a new line, numbered.

Question: {question}
Sub-questions:"""

    response = chat(
        model=llm_name, messages=[{"role": "user", "content": prompt}], temperature=0
    )

    raw = response.choices[0].message.content.strip()

    sub_questions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
            sub_questions.append(line.split(". ", 1)[-1])

    return sub_questions if sub_questions else [question]
