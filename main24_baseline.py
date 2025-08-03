from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import os
import gameof24_baseline as gameof24

load_dotenv()

NUM_SAMPLES = 5

API_KEYS = [
    os.getenv("OPENAI_API_KEY_1"),
    os.getenv("OPENAI_API_KEY_2"),
    os.getenv("OPENAI_API_KEY_3"),
    os.getenv("OPENAI_API_KEY_4")
]

def make_client(api_key):
    return OpenAI(api_key=api_key)

openai_clients = [make_client(k) for k in API_KEYS if k]
if not openai_clients:
    raise RuntimeError("No OpenAI API keys found!")

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

    current_client = openai_clients[i % len(openai_clients)]

    # Pass numbers as string, as expected by your baseline
    numbers_str = " ".join(map(str, numbers))
    content, tokens, (is_correct, message) = gameof24.solve24_oneshot(numbers_str, current_client)

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
