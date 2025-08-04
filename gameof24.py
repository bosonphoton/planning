import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# === SETUP OPENAI ===
# client = OpenAI(api_key="")

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


def action(current_state, current_tree, goal_state, client: OpenAI, model_name: str) -> Tuple[str, Dict[str, int]]:
    """Determine the next action (drilldown, solve, backtrack) based on the current state and tree."""
    current_tree_str = current_tree if isinstance(current_tree, str) else str(current_tree)

    prompt_action = f"""The current state is {current_state}.
    The current tree is {current_tree_str}.
    The goal state is {goal_state}.

    Choose from one of the actions below for the current state to get closer to the goal state:

    - drilldown: break the current state into even smaller more manageable subtasks (by combining two numbers with +, -, *, or /, but division must result in an integer)
    - solve: solve the task directly if it is simple enough
    - backtrack: the current path of decomposition or attempted solution is unproductive, it can return to a previous step and try a different approach to solving

    Return only the name of the action and nothing else.
    """
    content, tokens = call_llm(prompt_action, client, model_name)
    act = content.strip().lower().split()[0].strip(":.")  # robust to punctuation
    return act, tokens


def solve(current_state, goal_state, client: OpenAI, model_name: str) -> Tuple[str, Dict[str, int]]:
    """Ask the LLM to solve the current task directly."""
    prompt_solve = f""" 
    Here is the goal state: {goal_state}.
    Here is the current task: {current_state}.

    The solution for this current task will be used in part to reach the goal state.
    Solve the current task directly. Use only integer arithmetic.

    Respond with the solution or 'no solution'.
    """
    content, tokens = call_llm(prompt_solve, client, model_name)
    return content, tokens


def is_solved(numbers: List[int], target: int = 24) -> bool:
    """Solved iff a single value remains and it equals the target."""
    return len(numbers) == 1 and numbers[0] == target


def canonical_state(numbers: List[int]) -> str:
    """Canonical string for deduping states."""
    return str(sorted(numbers))


def recursive_24game_agent(
    current_state: List[int],
    current_path: List[str],
    goal_state: int,
    tree: List[str],
    visited: set,
    stats: Dict[str, int],
    client: OpenAI,
    model_name: str,
    depth: int = 0,
    max_depth: int = 5,
    token_list: List[Dict[str, int]] = None,
) -> Tuple[Optional[List[str]], List[Dict[str, int]]]:
    """Recursive agent to solve the 24 game using LLMs, tracking token usage."""
    if token_list is None:
        token_list = []

    indent = "  " * depth
    print(f"{indent}Current numbers: {current_state}, Path: {current_path}")

    if is_solved(current_state, goal_state):
        print(f"{indent}Solved! Path: {current_path}")
        return current_path, token_list

    state_key = canonical_state(current_state)
    if state_key in visited:
        print(f"{indent}Cycle detected, backtracking.")
        return None, token_list
    if depth >= max_depth:
        print(f"{indent}Max recursion depth reached, backtracking.")
        return None, token_list
    visited.add(state_key)

    chosen_action, tokens = action(current_state, tree, goal_state, client, model_name)
    token_list.append(tokens)
    print(f"{indent}LM chose action: {chosen_action} | tokens: {tokens}")

    if chosen_action == "solve":
        stats["solve"] = stats.get("solve", 0) + 1
        solution, tokens2 = solve(current_state, goal_state, client, model_name)
        token_list.append(tokens2)
        print(f"{indent}LM returned solution: {solution} | tokens: {tokens2}")
        if solution and "no solution" not in solution.lower():
            return current_path + [f"solve: {solution}"], token_list
        else:
            return None, token_list

    elif chosen_action == "drilldown":
        stats["drilldown"] = stats.get("drilldown", 0) + 1
        ops = [
            ('+', lambda x, y: x + y),
            ('-', lambda x, y: x - y),
            ('*', lambda x, y: x * y),
            ('/', lambda x, y: x // y if y != 0 and x % y == 0 else None),
        ]
        n = len(current_state)
        found_any = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for op_name, op in ops:
                    try:
                        new_num = op(current_state[i], current_state[j])
                        if new_num is None or abs(new_num) > 1e6:
                            continue
                    except Exception:
                        continue
                    if not isinstance(new_num, int):
                        continue
                    new_numbers = [current_state[k] for k in range(n) if k != i and k != j] + [new_num]
                    new_path = current_path + [f"({current_state[i]} {op_name} {current_state[j]}) -> {new_num}"]
                    tree.append(f"{current_state} => {new_numbers}")
                    result_path, _ = recursive_24game_agent(
                        new_numbers, new_path, goal_state, tree, visited, stats,
                        client, model_name, depth + 1, max_depth, token_list  # pass SAME token_list
                    )
                    if result_path:
                        return result_path, token_list
                    found_any = True
        if not found_any:
            print(f"{indent}No further decompositions possible, backtracking.")
        return None, token_list

    elif chosen_action == "backtrack":
        stats["backtrack"] = stats.get("backtrack", 0) + 1
        print(f"{indent}Backtracking.")
        return None, token_list

    else:
        print(f"{indent}Unknown action or LM output: '{chosen_action}', backtracking.")
        return None, token_list


def _sum_tokens(token_log: List[Dict[str, int]]) -> int:
    """Sum total tokens with a safe fallback when total_tokens is absent."""
    total = 0
    for t in token_log:
        if not t:
            continue
        if isinstance(t.get("total_tokens"), int):
            total += t["total_tokens"]
        else:
            total += t.get("prompt_tokens", 0) + t.get("completion_tokens", 0)
    return total


def play_24game_with_llm(numbers: List[int], stats: Dict[str, int], model_name: str, client: OpenAI, goal: int = 24):
    """Play the 24 game with the given numbers using the LLM agent."""
    print(f"\n=== Solving 24 Game: {numbers} ===")
    path, token_list = recursive_24game_agent(
        current_state=numbers,
        current_path=[],
        goal_state=goal,
        tree=[],
        visited=set(),
        stats=stats,
        client=client,
        model_name=model_name,
        depth=0,
        max_depth=5,
        token_list=[],
    )
    total_tokens = _sum_tokens(token_list)
    success = bool(path)

    if success:
        print("\n--- Solution Path ---")
        for step in path:
            print(step)
    else:
        print("\nNo solution found.")
    print(f"Tokens used for this problem: {total_tokens}\n")

    return success, total_tokens, token_list


if __name__ == "__main__":
    test_problems = [
        [4, 4, 10, 10],
        [1, 5, 5, 5],
        [8, 8, 3, 3],
        [3, 3, 8, 8],
        [1, 3, 4, 6],
    ]
    stats = {"solve": 0, "drilldown": 0, "backtrack": 0}

    # Instantiate client (or exit early if missing)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    client = OpenAI(api_key=api_key)

    solved_count = 0
    all_token_info = []

    for nums in test_problems:
        success, tokens_used, token_log = play_24game_with_llm(nums, stats, client)
        solved_count += int(success)
        all_token_info.append({"numbers": nums, "solved": success, "tokens_used": tokens_used})

    print(f"\nSolved {solved_count}/{len(test_problems)}. Accuracy: {solved_count/len(test_problems):.2f}")
    total_tokens = sum(e["tokens_used"] for e in all_token_info)
    print(f"Total tokens used: {total_tokens}")
