from datasets import load_dataset
from typing import Dict
from openai import OpenAI
import gameof24


NUM_SAMPLES = 200

API_KEYS = [
    "",
    "",
    "",
    ""
]

def make_client(api_key):
    return OpenAI(api_key=api_key)

openai_clients = [make_client(k) for k in API_KEYS]

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

    current_client = openai_clients[i % len(openai_clients)]

    if gameof24.play_24game_with_llm(numbers=numbers, stats=chosen_action_count, client=current_client):
        solved_count += 1
        difficulties_solved[difficulty] += 1
    all_difficulies[difficulty] += 1

print(f"\nSolved {solved_count}/{NUM_SAMPLES}. Accuracy: {solved_count/NUM_SAMPLES:.2f}")
print(chosen_action_count)

print(f"Easy solved: {difficulties_solved['easy']} / {all_difficulies['easy']}")
print(f"Medium solved: {difficulties_solved['medium']} / {all_difficulies['medium']}")
print(f"Hard solved: {difficulties_solved['hard']} / {all_difficulies['hard']}")

