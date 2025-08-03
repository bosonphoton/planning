from datasets import load_dataset
from typing import Dict

from openai import OpenAI
from together import Together

import gameof24
import os
from dotenv import load_dotenv


import argparse


load_dotenv()  # Loads the .env file

parser = argparse.ArgumentParser(description="Financial IF Evaluation")
parser.add_argument("--provider", default="openai",
                       help="Model provider, either API based or litellm/vllm")
parser.add_argument("--model_name", default="gpt-3.5-turbo",
                       help="model name as per the provider")
parser.add_argument("--num_samples", type=int, default=200,
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

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nlile/24-game")

# turns out all examples from this dataset are solvable anyway
ds = ds.filter(lambda x: x["solvable"])

def get_difficulty(example):
    sr = example["solved_rate"]
    if sr > 0.9: return "easy"
    elif sr > 0.6: return "medium"
    return "hard"

ds = ds.map(lambda x: {"difficulty": get_difficulty(x)})


solved_count = 0
difficulties_solved: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}
all_difficulies: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}
chosen_action_count: Dict[str, int] = {'solve': 0, 'drilldown': 0, 'backtrack': 0}

print(f"\n\n***** {NUM_SAMPLES} samples *****")

for i in range(NUM_SAMPLES):
    print(f"\n\n==== Problem {i} ====")
    problem = ds["train"][i]
    difficulty = problem['difficulty']
    numbers = problem['numbers']

    current_client = provider_clients[i % len(provider_clients)]

    if gameof24.play_24game_with_llm(numbers=numbers, stats=chosen_action_count, model_name=MODEL_NAME, client=current_client):
        solved_count += 1
        difficulties_solved[difficulty] += 1
    all_difficulies[difficulty] += 1

print(f"\nSolved {solved_count}/{NUM_SAMPLES}. Accuracy: {solved_count/NUM_SAMPLES:.2f}")
print(chosen_action_count)

print(f"Easy solved: {difficulties_solved['easy']} / {all_difficulies['easy']}")
print(f"Medium solved: {difficulties_solved['medium']} / {all_difficulies['medium']}")
print(f"Hard solved: {difficulties_solved['hard']} / {all_difficulies['hard']}")

