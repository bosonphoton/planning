import re
import sympy
from typing import Tuple, Dict
from openai import OpenAI

def call_llm(prompt: str, client: OpenAI, model_name: str) -> Tuple[str, Dict[str, int]]:
    """Basic function to call the OpenAI API with a given prompt and return (content, tokens)."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=64,
    )
    content = response.choices[0].message.content.strip()
    usage = getattr(response, "usage", {}) or {}
    prompt_t = getattr(usage, "prompt_tokens", 0)
    compl_t  = getattr(usage, "completion_tokens", 0)
    total_t  = getattr(usage, "total_tokens", prompt_t + compl_t)
    tokens = {"prompt_tokens": prompt_t, "completion_tokens": compl_t, "total_tokens": total_t}
    return content, tokens

def evaluate_24_solution(input_numbers: str, llm_output: str) -> Tuple[bool, str]:
    """
    Checks if the LLM output is a valid solution for the 24 Game puzzle.
    Returns (True/False, message).
    """
    last_line = llm_output.strip().split('\n')[-1].strip()
    # NEW: Remove 'Answer:' or similar prefixes if present
    if last_line.lower().startswith("answer:"):
        last_line = last_line[len("answer:"):].strip()
    expr = last_line.split('=')[0].strip()  # Handles outputs like "... = 24" and "..."

    # Check number usage (must match input, no more, no less)
    input_nums = sorted(re.findall(r'\d+', input_numbers))
    expr_nums = sorted(re.findall(r'\d+', expr))
    if input_nums != expr_nums:
        return False, f"Numbers used do not match input: expected {input_nums}, got {expr_nums}"

    # Evaluate with sympy and check if result is 24
    try:
        result = sympy.simplify(expr)
        if result == 24:
            return True, "Correct solution."
        else:
            return False, f"Expression does not evaluate to 24, got {result}."
    except Exception as e:
        return False, f"Could not parse expression: {e}"


def solve24_oneshot(input_numbers: str, client: OpenAI, model_name: str) -> Tuple[str, Dict[str, int], Tuple[bool, str]]:
    """Ask the LLM to solve the current task directly, and evaluate correctness."""

    prompt_solve = f"""Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
    Input: 4 4 6 8
    Answer: (4 + 8) * (6 - 4) = 24
    Input: 2 9 10 12
    Answer: 2 * 12 * (10 - 9) = 24
    Input: 4 9 10 13
    Answer: (13 - 9) * (10 - 4) = 24
    Input: 1 4 8 8
    Answer: (8 / 4 + 1) * 8 = 24
    Input: 5 5 5 9
    Answer: 5 + 5 + 5 + 9 = 24
    Input: {input_numbers}
    """

    content, tokens = call_llm(prompt_solve, client, model_name)
    is_correct, message = evaluate_24_solution(input_numbers, content)
    return content, tokens, (is_correct, message)



