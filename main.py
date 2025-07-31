from datasets import load_dataset

import gameof24


NUM_SAMPLES = 3

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nlile/24-game")

# turns out all examples from this dataset are solvable anyway
ds = ds.filter(lambda x: x["solvable"])

print(ds)

# first 5
print(ds["train"][:5])


solved_count = 0
for i in range(NUM_SAMPLES):
    print(f"==== Problem {i} ====")
    problem = ds["train"][i]
    numbers = problem['numbers']
    if gameof24.play_24game_with_llm(numbers=numbers):
        solved_count += 1

print(f"\nSolved {solved_count}/{NUM_SAMPLES}. Accuracy: {solved_count/NUM_SAMPLES:.2f}")


