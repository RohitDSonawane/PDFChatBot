"""
llm.py - OpenRouter LLM API call with exponential-backoff retry logic.
"""

import os
import time

from openai import OpenAI, RateLimitError

from config import OPENROUTER_MODEL, LLM_MAX_RETRIES

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided PDF context. If the answer is not in the context, say so clearly."
)


def get_answer(context: str, question: str) -> str:
    """
    Send the retrieved context and user question to OpenRouter's chat API
    and return the model's answer.

    Retries up to LLM_MAX_RETRIES times with exponential backoff (2s, 4s, 8s)
    on 429 rate-limit errors before re-raising.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context from PDF:\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))  # 2s -> 4s -> 8s
            else:
                raise
