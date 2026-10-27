"""Q-Learning core factored out of 'Zeniabe Final V.ipynb'.

The six state-action transitions, reward mapping, epsilon-greedy tie handling,
Bellman update, 500-episode reset convention and reward accumulation follow
that notebook. Unlike the notebook's global Q table, Q and random state are
passed explicitly so experiments and browser sessions stay independent.
"""

import random
from typing import Dict, List

STATES = ("Cold", "Comfortable", "Hot")
HEATER_ON = 0
HEATER_OFF = 1
ACTIONS = (HEATER_ON, HEATER_OFF)
INITIAL_STATE = "Cold"
EXPECTED_POLICY = {"Cold": HEATER_ON, "Comfortable": HEATER_OFF, "Hot": HEATER_OFF}
ACTION_NAMES = {HEATER_ON: "Heater ON", HEATER_OFF: "Heater OFF"}
# Infinite-horizon, gamma=0.9 theoretical solution for THIS deterministic MDP.
# Different gamma values require recomputing it with theoretical_q().


def transition(state: str, action: int) -> str:
    """The transition function copied from the user's notebook (same logic)."""
    if state not in STATES:
        raise ValueError("Invalid state")
    if action not in ACTIONS:
        raise ValueError("Invalid action")

    if state == "Cold":
        if action == HEATER_ON:
            return "Comfortable"
        else:
            return "Cold"
    elif state == "Comfortable":
        if action == HEATER_ON:
            return "Hot"
        else:
            return "Comfortable"
    elif state == "Hot":
        if action == HEATER_ON:
            return "Hot"
        else:
            return "Comfortable"
    raise ValueError("Invalid state")


def reward_function(next_state: str) -> int:
    """Exact reward lookup used in the notebook: reward of the NEXT state."""
    if next_state == "Comfortable":
        return 10
    elif next_state == "Hot":
        return -5
    elif next_state == "Cold":
        return -1
    else:
        raise ValueError("Invalid state")


def blank_q() -> Dict[str, Dict[int, float]]:
    return {s: {a: 0.0 for a in ACTIONS} for s in STATES}


def greedy_action(state: str, q: dict) -> int:
    if state not in STATES:
        raise ValueError("Invalid state")
    # Python max ties favor action 0; notebook evaluation uses the same max.
    return max(ACTIONS, key=lambda a: q[state][a])


def choose_action(state: str, q: dict, epsilon: float, rng: random.Random) -> int:
    """Same epsilon-greedy decision + randomly resolved ties as notebook."""
    if state not in STATES:
        raise ValueError("Invalid state")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    best_q = max(q[state].values())
    best_actions = []
    for action in ACTIONS:
        if q[state][action] == best_q:
            best_actions.append(action)
    return rng.choice(best_actions)


def update_q(state: str, action: int, reward: int, next_state: str,
             q: dict, alpha: float, gamma: float) -> dict:
    """Bellman update from notebook, additionally returns numerical trace."""
    old_q = q[state][action]
    best_next_q = max(q[next_state].values())
    new_q = old_q + alpha * (reward + gamma * best_next_q - old_q)
    q[state][action] = new_q
    return {
        "state": state, "action": action, "next_state": next_state,
        "reward": reward, "old_q": old_q, "best_next_q": best_next_q,
        "new_q": new_q, "alpha": alpha, "gamma": gamma,
    }


def train(episodes: int = 500, steps_per_episode: int = 20,
          epsilon: float = 0.1, alpha: float = 0.1,
          gamma: float = 0.9, seed: int = 42,
          epsilon_decay: float = 0.995,
          epsilon_min: float = 0.01) -> dict:
    """Notebook-equivalent training from zero Q, resetting state to Cold only.

    Episodes are time truncations, not terminal states: the Bellman backup at
    the last step still bootstraps from max Q(next_state), as in the notebook.
    """
    if episodes < 1 or steps_per_episode < 1:
        raise ValueError("episodes and steps_per_episode must be positive")
    if not (0 < alpha <= 1 and 0 <= gamma < 1):
        raise ValueError("alpha must be (0,1] and gamma must be [0,1)")
    q = blank_q()
    rng = random.Random(seed)
    current_epsilon = epsilon
    episode_rewards: List[int] = []
    cumulative_rewards: List[int] = []
    cumulative_total = 0
    last_update = None
    for _ in range(episodes):
        state = INITIAL_STATE
        total_reward = 0
        for _ in range(steps_per_episode):
            action = choose_action(state, q, current_epsilon, rng)
            next_state = transition(state, action)
            reward = reward_function(next_state)
            last_update = update_q(state, action, reward, next_state, q, alpha, gamma)
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
        cumulative_total += total_reward
        cumulative_rewards.append(cumulative_total)
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
    return {
        "q": q,
        "episode_rewards": episode_rewards,
        "cumulative_rewards": cumulative_rewards,
        "last_update": last_update,
        "config": {"episodes": episodes, "steps_per_episode": steps_per_episode,
                   "epsilon": epsilon, "alpha": alpha, "gamma": gamma, "seed": seed},
    }


def evaluate(q: dict, start_state: str = INITIAL_STATE, steps: int = 20) -> dict:
    """No learning and no epsilon exploration during evaluation."""
    if start_state not in STATES:
        raise ValueError("Invalid state")
    state = start_state
    total = 0
    timeline = []
    for step in range(1, steps + 1):
        action = greedy_action(state, q)
        next_state = transition(state, action)
        reward = reward_function(next_state)
        total += reward
        timeline.append({"step": step, "state": state, "action": ACTION_NAMES[action],
                         "next_state": next_state, "reward": reward, "total": total})
        state = next_state
    return {"total": total, "timeline": timeline, "maximum": steps * 10,
            "start_state": start_state}


def theoretical_q(gamma: float = 0.9) -> dict:
    """Compute exact optimal Q for these deterministic transitions.

    From every state, the optimal next action yields +10 forever, so V*=10/(1-gamma).
    This is independent of the 20-step display truncation.
    """
    optimal_value = 10 / (1 - gamma)
    return {
        s: {a: reward_function(transition(s, a)) + gamma * optimal_value
            for a in ACTIONS} for s in STATES
    }


def q_mae(q: dict, gamma: float = 0.9) -> float:
    target = theoretical_q(gamma)
    return sum(abs(q[s][a] - target[s][a]) for s in STATES for a in ACTIONS) / 6


def policy_correct_count(q: dict) -> int:
    return sum(greedy_action(s, q) == a for s, a in EXPECTED_POLICY.items())
