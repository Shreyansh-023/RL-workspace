import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (from lab)
# -----------------------------
NUM_ARMS = 10          # 10-armed bandit
STEPS = 10000          # 10000 iterations
C = 2                  # UCB exploration constant

np.random.seed(42)

# -----------------------------
# True reward means (unknown)
# -----------------------------
true_means = np.random.normal(0, 1, NUM_ARMS)
optimal_arm = np.argmax(true_means)

# -----------------------------
# Initialization
# -----------------------------
Q = np.zeros(NUM_ARMS)         # Estimated value of each arm
N = np.zeros(NUM_ARMS)         # Number of times each arm selected

rewards = np.zeros(STEPS)
optimal_action_count = np.zeros(STEPS)

# -----------------------------
# Initialization step:
# Select each arm once
# -----------------------------
t = 0
for arm in range(NUM_ARMS):
    reward = np.random.normal(true_means[arm], 1)

    N[arm] += 1
    Q[arm] += (reward - Q[arm]) / N[arm]  # sample average update

    rewards[t] = reward
    optimal_action_count[t] = (arm == optimal_arm)
    t += 1

# -----------------------------
# Main UCB Loop
# -----------------------------
for t in range(NUM_ARMS, STEPS):

    # UCB action selection
    ucb_values = Q + C * np.sqrt(np.log(t + 1) / N)
    action = np.argmax(ucb_values)

    # Take action and get reward
    reward = np.random.normal(true_means[action], 1)

    # Update estimates (incremental mean)
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]

    # Store metrics
    rewards[t] = reward
    optimal_action_count[t] = (action == optimal_arm)

# -----------------------------
# Performance Metrics
# -----------------------------
average_reward = np.cumsum(rewards) / np.arange(1, STEPS + 1)
optimal_action_percentage = np.cumsum(optimal_action_count) / np.arange(1, STEPS + 1)

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(average_reward)
plt.xlabel("Time Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward vs Time")

plt.subplot(1, 2, 2)
plt.plot(optimal_action_percentage * 100)
plt.xlabel("Time Steps")
plt.ylabel("% Optimal Action")
plt.title("Optimal Action Percentage vs Time")

plt.tight_layout()
plt.show()
