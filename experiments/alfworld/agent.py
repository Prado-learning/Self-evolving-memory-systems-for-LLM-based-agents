"""ALFWorld agent: uses LLM + memory to interact with ALFWorld environment."""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert household robot in a text adventure. You solve tasks efficiently.

Key strategies by task type:
- COOL then place: find the object, take it, go to fridge 1, cool it, go to target, put it.
- HEAT then place: find the object, take it, go to microwave 1, heat it, go to target, put it.
- CLEAN then place: find the object, take it, go to sinkbasin 1, clean it, go to target, put it.
- PICK AND PLACE: find the object, take it, go to target, put it.
- EXAMINE with light: find the object, take it, find a desklamp, use desklamp.
- PICK TWO objects: do pick-and-place twice, one object at a time.

Where to search for objects:
- countertop, diningtable, shelf, drawer, cabinet, fridge, sinkbasin, stoveburner, desk, sidetable, dresser, coffeetable, bed, armchair, sofa, toilet, bathtubbasin, garbagecan

CRITICAL RULES:
1. You MUST find the EXACT object mentioned in the task. Read the task carefully.
2. Search multiple locations systematically. If the object is not at the first location, try the next.
3. Always reply with EXACTLY one action from the admissible commands list. No extra text.

Here is an example of a successful task:

Task: cool some bread and put it in countertop.
> go to countertop 1
On the countertop 1, you see a bread 1, a knife 1.
> take bread 1 from countertop 1
You pick up the bread 1 from the countertop 1.
> go to fridge 1
You arrive at fridge 1. The fridge 1 is closed.
> cool bread 1 with fridge 1
You cool the bread 1 using the fridge 1.
> go to countertop 2
You arrive at countertop 2.
> put bread 1 in/on countertop 2
You put the bread 1 in/on the countertop 2.
DONE

Here is another example:

Task: examine the pencil with the desklamp.
> go to desk 1
On the desk 1, you see a pencil 1, a book 1.
> take pencil 1 from desk 1
You pick up the pencil 1 from the desk 1.
> go to desklamp 1
You arrive at desklamp 1.
> use desklamp 1
You turn on the desklamp 1. You examine the pencil 1 under the desklamp 1.
DONE"""


def format_memories(memories: List[Dict], max_memories: int = 3) -> str:
    if not memories:
        return ""
    lines = ["Here are relevant past experiences:"]
    for i, mem in enumerate(memories[:max_memories]):
        lines.append(f"\nExperience {i+1} (task: {mem.get('task_type', '?')}):")
        exp_text = mem["experience"]
        if len(exp_text) > 500:
            exp_text = exp_text[:500] + "..."
        lines.append(exp_text)
    return "\n".join(lines)


def build_prompt(
    goal: str,
    obs: str,
    admissible_commands: List[str],
    memories: Optional[List[Dict]] = None,
    history: Optional[List[str]] = None,
) -> str:
    parts = [f"Task: {goal}"]

    if memories:
        parts.append(format_memories(memories))

    if history:
        parts.append("\nRecent actions:")
        for h in history[-10:]:  # last 10 actions
            parts.append(f"  {h}")

    parts.append(f"\nCurrent observation: {obs}")
    parts.append(f"\nAdmissible commands: {admissible_commands}")
    parts.append("\nChoose the best next action. Reply with ONLY the action text, exactly as listed.")

    return "\n".join(parts)


def parse_action(response: str, admissible_commands: List[str]) -> str:
    response = response.strip().strip('"').strip("'")

    # Exact match
    if response in admissible_commands:
        return response

    # Case-insensitive match
    lower_map = {cmd.lower(): cmd for cmd in admissible_commands}
    if response.lower() in lower_map:
        return lower_map[response.lower()]

    # Substring match: find the command that appears in response
    for cmd in admissible_commands:
        if cmd in response:
            return cmd

    # Fuzzy: find best overlap
    best_cmd = admissible_commands[0]
    best_score = 0
    for cmd in admissible_commands:
        words = set(cmd.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(words & resp_words)
        if overlap > best_score:
            best_score = overlap
            best_cmd = cmd
    return best_cmd


class ALFWorldAgent:
    def __init__(self, llm_client, max_tokens: int = 64):
        self.llm = llm_client
        self.max_tokens = max_tokens

    def act(
        self,
        goal: str,
        obs: str,
        admissible_commands: List[str],
        memories: Optional[List[Dict]] = None,
        history: Optional[List[str]] = None,
    ) -> str:
        prompt = build_prompt(goal, obs, admissible_commands, memories, history)
        response = self.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return parse_action(response, admissible_commands)
