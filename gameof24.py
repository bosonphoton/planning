import os
from typing import List
from openai import OpenAI
from typing import Dict

# === SETUP OPENAI ===
client = OpenAI(api_key="")

def call_llm(prompt):
    """basic function to call the OpenAI API with a given prompt"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=64,
    )
    content = response.choices[0].message.content.strip()
    return content

def action(current_state, current_tree, goal_state):
    """Determine the next action (drill,solve,backtrack) based on the current state and tree"""
    
    if not isinstance(current_tree, str):
        current_tree_str = str(current_tree)
    else:
        current_tree_str = current_tree

    prompt_action = f"""The current state is {current_state}.
    The current tree is {current_tree_str}.
    The goal state is {goal_state}.

    Choose from one of the actions below for the current state to get closer to the goal state:

    - drilldown: break the current state into even smaller more manageable subtasks (by combining two numbers with +, -, *, or /, but division must result in an integer)
    - solve: solve the task directly if it is simple enough
    - backtrack: the current path of decomposition or attempted solution is unproductive, it can return to a previous step and try a different approach to solving

    Return only the name of the action and nothing else.
    """
    return call_llm(prompt_action).lower()

def solve(current_state, goal_state):
    """Solve the current task directly using the LLM"""
    
    prompt_solve = f""" 
    Here is the goal state: {goal_state}.
    Here is the current task: {current_state}.

    The solution for this current task will be used in part to reach the goal state.
    Solve the current task directly. Use only integer arithmetic.

    Respond with the solution or 'no solution'.
    """
    return call_llm(prompt_solve)

def is_solved(numbers: List[int], target=24) -> bool:
    """Check if the current numbers contain the target value (default is 24)"""
    return any(n == target for n in numbers)

def canonical_state(numbers: List[int]) -> str:
    """Return a canonical string representation of the current state of numbers"""
    return str(sorted(numbers))

def recursive_24game_agent(
    current_state: List[int], 
    current_path: List[str],
    goal_state: int,
    tree: List[str],
    visited: set,
    stats: Dict[str, int],
    depth=0,
    max_depth=5
    ):
    """Recursive agent to solve the 24 game using LLMs"""
    indent = "  " * depth
    print(f"{indent}Current numbers: {current_state}, Path: {current_path}")

    if is_solved(current_state, goal_state):
        print(f"{indent}Solved! Path: {current_path}")
        return current_path

    state_key = canonical_state(current_state)
    if state_key in visited:
        print(f"{indent}Cycle detected, backtracking.")
        return None
    if depth > max_depth:
        print(f"{indent}Max recursion depth reached, backtracking.")
        return None
    visited.add(state_key)

    chosen_action = action(current_state, tree, goal_state)
    print(f"{indent}LM chose action: {chosen_action}")

    if chosen_action == "solve":
        stats["solve"] += 1
        solution = solve(current_state, goal_state)
        print(f"{indent}LM returned solution: {solution}")
        if solution and "no solution" not in solution.lower():
            return current_path + [f"solve: {solution}"]
        else:
            return None

    elif chosen_action == "drilldown":
        stats["drilldown"] += 1
        ops = [('+', lambda x, y: x + y), 
               ('-', lambda x, y: x - y),
               ('*', lambda x, y: x * y), 
               ('/', lambda x, y: x // y if y != 0 and x % y == 0 else None)]
        n = len(current_state)
        found_any = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op_name, op in ops:
                    try:
                        new_num = op(current_state[i], current_state[j])
                        # Only accept integer result (not None)
                        if new_num is None or abs(new_num) > 1e6:
                            continue
                    except Exception:
                        continue
                    # Ensure new_num is integer
                    if not isinstance(new_num, int):
                        continue
                    new_numbers = [current_state[k] for k in range(n) if k != i and k != j] + [new_num]
                    new_path = current_path + [f"({current_state[i]} {op_name} {current_state[j]}) -> {new_num}"]
                    tree.append(f"{current_state} => {new_numbers}")
                    result = recursive_24game_agent(new_numbers, new_path, goal_state, tree, visited, stats, depth+1, max_depth)
                    if result:
                        return result
                    found_any = True
        if not found_any:
            print(f"{indent}No further decompositions possible, backtracking.")
        return None

    elif chosen_action == "backtrack":
        stats["backtrack"] += 1
        print(f"{indent}Backtracking.")
        return None
    else:
        print(f"{indent}Unknown action or LM output: '{chosen_action}', backtracking.")
        return None

def play_24game_with_llm(numbers: List[int], stats: Dict[str, int], goal: int = 24):
    """Play the 24 game with the given numbers using the LLM agent"""
    print(f"\n=== Solving 24 Game: {numbers} ===")
    path = recursive_24game_agent(
        current_state=numbers, 
        current_path=[], 
        goal_state=goal,
        tree=[],
        visited=set(),
        stats=stats,
        depth=0,
        max_depth=5
    )
    if path:
        print("\n--- Solution Path ---")
        for step in path:
            print(step)
        return True
    else:
        print("\nNo solution found.")
        return False

if __name__ == "__main__":
    test_problems = [
        [4, 4, 10, 10],
        [1, 5, 5, 5],
        [8, 8, 3, 3],
        [3, 3, 8, 8],
        [1, 3, 4, 6]
    ]
    solved_count = 0
    for nums in test_problems:
        if play_24game_with_llm(nums):
            solved_count += 1
    print(f"\nSolved {solved_count}/{len(test_problems)}. Accuracy: {solved_count/len(test_problems):.2f}")
