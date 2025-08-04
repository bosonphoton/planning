from datasets import load_dataset
from typing import Dict

from openai import OpenAI
from together import Together

from dotenv import load_dotenv
import os
import gameof24_baseline as gameof24

import argparse

load_dotenv()

parser = argparse.ArgumentParser(description="Baseline Solution for 24 Main")
parser.add_argument("--provider", default="openai",
                       help="Model provider, either API based or litellm/vllm")
parser.add_argument("--model_name", default="gpt-3.5-turbo",
                       help="model name as per the provider")
parser.add_argument("--num_samples", type=int, default=5,
                       help="number of samples to evaluate on")
args = parser.parse_args()


NUM_SAMPLES = args.num_samples
PROVIDER = args.provider
MODEL_NAME = args.model_name

API_KEYS =[]

if PROVIDER == "openai":
    API_KEYS = [
        os.getenv("OPENAI_API_KEY_1"),
        os.getenv("OPENAI_API_KEY_2"),
        os.getenv("OPENAI_API_KEY_3"),
        os.getenv("OPENAI_API_KEY_4")
    ]
elif PROVIDER == "together":
    API_KEYS = [
        os.getenv("TOGETHER_API_KEY_1"),
        os.getenv("TOGETHER_API_KEY_2"),
        os.getenv("TOGETHER_API_KEY_3"),
        os.getenv("TOGETHER_API_KEY_4")
    ]

def make_client(api_key,provider):
    if provider == "openai":
        return OpenAI(api_key=api_key)
    else:
        return Together(api_key=api_key)

provider_clients = [make_client(k,PROVIDER) for k in API_KEYS]


if not provider_clients:
    raise RuntimeError("No Provider keys found!")

ds = load_dataset("nlile/24-game")
ds = ds.filter(lambda x: x["solvable"])

def get_difficulty(example):
    sr = example["solved_rate"]
    if sr > 0.9: return "easy"
    elif sr > 0.6: return "medium"
    return "hard"

ds = ds.map(lambda x: {"difficulty": get_difficulty(x)})

solved_count = 0
difficulties_solved = {'easy': 0, 'medium': 0, 'hard': 0}
all_difficulties = {'easy': 0, 'medium': 0, 'hard': 0}

print(f"\n\n***** {NUM_SAMPLES} samples *****")

for i in range(NUM_SAMPLES):
    print(f"\n\n==== Problem {i} ====")
    problem = ds["train"][i]
    difficulty = problem['difficulty']
    numbers = problem['numbers']
    all_difficulties[difficulty] += 1

    current_client = provider_clients[i % len(provider_clients)]

    # Pass numbers as string, as expected by your baseline
    numbers_str = " ".join(map(str, numbers))
    content, tokens, (is_correct, message) = gameof24.solve24_oneshot(numbers_str, current_client, MODEL_NAME)

    print(f"Numbers: {numbers} | Difficulty: {difficulty}")
    print(f"LLM Output: {content}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'} | Detail: {message}")
    print(f"Token usage: {tokens}")

    if is_correct:
        solved_count += 1
        difficulties_solved[difficulty] += 1

print(f"\nSolved {solved_count}/{NUM_SAMPLES}. Accuracy: {solved_count/NUM_SAMPLES:.2f}")
for diff in ['easy', 'medium', 'hard']:
    print(f"{diff.capitalize()} solved: {difficulties_solved[diff]} / {all_difficulties[diff]}")
