# Assignment 1: Smart Room Temperature Control using Q-learning
# A simple Python version of the same code in the notebook.

# Assignment 1: Smart Room Temperature Control
# Phase 1: Environment and Q-table
# States and actions
states = ["Cold", "Comfortable", "Hot"]
HEATER_ON = 0
HEATER_OFF = 1
actions = [HEATER_ON, HEATER_OFF]

# Assignment settings
initial_state = "Cold"
alpha = 0.1
gamma = 0.9
episodes = 500

print("States:", states)
print("Actions: Heater ON, Heater OFF")

# Transition function
def transition(state, action):
    if state not in states:
        raise ValueError("Invalid state")
    if action not in actions:
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

# Reward function
def reward_function(next_state):
    if next_state == "Comfortable":
        return 10
    elif next_state == "Hot":
        return -5
    elif next_state == "Cold":
        return -1
    else:
        raise ValueError("Invalid state")

# Test the environment
for state in states:
    for action in actions:
        next_state = transition(state, action)
        reward = reward_function(next_state)

        if action == HEATER_ON:
            action_name = "ON"
        else:
            action_name = "OFF"

        print(state, "+", action_name, "->", next_state, "Reward:", reward)

# Initialize the Q-table
Q = {}

for state in states:
    Q[state] = {}
    for action in actions:
        Q[state][action] = 0.0

print("Initial Q-table:")
for state in states:
    print(state, Q[state])

# Phase 2: Q-learning and training
import random

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
steps_per_episode = 100


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)

    if Q[state][HEATER_ON] > Q[state][HEATER_OFF]:
        return HEATER_ON
    else:
        return HEATER_OFF


def update_q(state, action, reward, next_state):
    old_q = Q[state][action]
    best_next_q = max(Q[next_state].values())
    Q[state][action] = old_q + alpha * (reward + gamma * best_next_q - old_q)

# Training for 500 episodes
random.seed(42)

episode_rewards = []
cumulative_rewards = []
cumulative_total = 0

for episode in range(episodes):
    state = initial_state
    total_reward = 0

    for step in range(steps_per_episode):
        action = choose_action(state)
        next_state = transition(state, action)
        reward = reward_function(next_state)

        update_q(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    episode_rewards.append(total_reward)
    cumulative_total += total_reward
    cumulative_rewards.append(cumulative_total)

    # Reduce random exploration after every episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training completed:", episodes, "episodes")

# Phase 3: Results
print("Final Q-table:")
print("State           Heater ON      Heater OFF")

for state in states:
    print(f"{state:<15} {Q[state][HEATER_ON]:<14.2f} {Q[state][HEATER_OFF]:.2f}")

print("\nBest action for each state:")
for state in states:
    if Q[state][HEATER_ON] > Q[state][HEATER_OFF]:
        best_action = "Heater ON"
    else:
        best_action = "Heater OFF"
    print(state, "->", best_action)

# Bellman equation: calculation by hand
first_update = 0 + alpha * (10 + gamma * 0 - 0)
print("Hand calculation:", first_update)

# Reward graphs
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, episodes + 1), episode_rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("reward_per_episode.png", bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(range(1, episodes + 1), cumulative_rewards)
plt.title("Cumulative Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.savefig("cumulative_reward_per_episode.png", bbox_inches="tight")
plt.show()

# Learning performance
first_100 = sum(episode_rewards[:100]) / 100
last_100 = sum(episode_rewards[-100:]) / 100

print("Average reward (first 100):", round(first_100, 2))
print("Average reward (last 100):", round(last_100, 2))
print("Total cumulative reward:", cumulative_rewards[-1])

if last_100 > first_100:
    print("Rewards improved during training.")
else:
    print("Rewards did not improve during training.")

print("Lower epsilon means fewer random heater actions near the end.")
print("This model uses fixed transitions, not real temperature readings.")

