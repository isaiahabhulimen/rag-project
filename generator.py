from config import llm_name
from llm_client import chat


def generate_answer(question, context):
    prompt = f"""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    Give a short answer with the important fact from the context. 
    Do not add information that is not in the context.

    answer:
    """

    response = chat(
        model=llm_name,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided context. Never add information that is not present in the context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    generated_answer = response.choices[0].message.content

    return generated_answer
