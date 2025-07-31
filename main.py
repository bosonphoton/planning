from datasets import load_dataset
from typing import Dict
import gameof24


NUM_SAMPLES = 5

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nlile/24-game")

# turns out all examples from this dataset are solvable anyway
ds = ds.filter(lambda x: x["solvable"])

# print(ds)

# first 5
# print(ds["train"][:5])


solved_count = 0
chosen_action_count: Dict[str, int] = {'solve': 0, 'drilldown': 0, 'backtrack': 0}

print(f"\n\n***** {NUM_SAMPLES} samples *****")
for i in range(NUM_SAMPLES):
    print(f"\n\n==== Problem {i} ====")
    problem = ds["train"][i]
    numbers = problem['numbers']
    if gameof24.play_24game_with_llm(numbers=numbers, stats=chosen_action_count):
        solved_count += 1

print(f"\nSolved {solved_count}/{NUM_SAMPLES}. Accuracy: {solved_count/NUM_SAMPLES:.2f}")
print(chosen_action_count)

