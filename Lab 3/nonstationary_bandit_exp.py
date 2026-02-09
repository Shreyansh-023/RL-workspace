"""
Exercise 2.5 (programming) - Nonstationary 10-armed Bandit Experiment

This script demonstrates the difficulties that sample-average methods have for nonstationary problems.
It compares two action-value methods:
1. Sample averages (incrementally computed)
2. Constant step-size parameter (alpha = 0.1)

Both methods use epsilon-greedy action selection (epsilon = 0.1).
The true action values q_*(a) perform independent random walks.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class NonstationaryBandit:
    def __init__(self, k=10, mu=0.0, sigma=1.0, walk_std=0.01):
        self.k = k
        self.q_true = np.ones(k) * mu
        self.sigma = sigma
        self.walk_std = walk_std

    def step(self):
        # Random walk for each action value
        self.q_true += np.random.normal(0, self.walk_std, self.k)

    def get_reward(self, action):
        # Reward is normal around current q_*(a)
        return np.random.normal(self.q_true[action], self.sigma)

    def reset(self):
        self.q_true[:] = 0.0


def run_bandit(method, steps=10000, runs=2000, k=10, epsilon=0.1, alpha=0.1, walk_std=0.01):
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))
    for run in range(runs):
        bandit = NonstationaryBandit(k=k, mu=0.0, sigma=1.0, walk_std=walk_std)
        Q = np.zeros(k)
        N = np.zeros(k)
        for t in range(steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else:
                action = np.argmax(Q)
            reward = bandit.get_reward(action)
            rewards[run, t] = reward
            optimal = np.argmax(bandit.q_true)
            optimal_actions[run, t] = (action == optimal)
            if method == 'sample_average':
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]
            elif method == 'constant_alpha':
                Q[action] += alpha * (reward - Q[action])
            bandit.step()
    return rewards, optimal_actions


def plot_results(avg_rewards_sa, opt_actions_sa, avg_rewards_ca, opt_actions_ca):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards_sa, label='Sample Average')
    plt.plot(avg_rewards_ca, label='Constant Step-size (alpha=0.1)')
    if len(plot_results.extra_rewards) > 0:
        plt.plot(plot_results.extra_rewards[0], label='Constant Step-size (alpha=0.01)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Average Reward')

    plt.subplot(1, 2, 2)
    plt.plot(opt_actions_sa * 100, label='Sample Average')
    plt.plot(opt_actions_ca * 100, label='Constant Step-size (alpha=0.1)')
    if len(plot_results.extra_opt) > 0:
        plt.plot(plot_results.extra_opt[0] * 100, label='Constant Step-size (alpha=0.01)')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.title('Optimal Action (%)')
    plt.tight_layout()
    plt.show()

# For extra method results
plot_results.extra_rewards = []
plot_results.extra_opt = []


def main():
    steps = 10000
    runs = 2000
    epsilon = 0.1
    alpha = 0.1
    walk_std = 0.01
    print('Running sample-average method...')
    rewards_sa, opt_sa = run_bandit('sample_average', steps, runs, epsilon=epsilon, alpha=alpha, walk_std=walk_std)
    print('Running constant-alpha method (alpha=0.1)...')
    rewards_ca, opt_ca = run_bandit('constant_alpha', steps, runs, epsilon=epsilon, alpha=alpha, walk_std=walk_std)
    print('Running constant-alpha method (alpha=0.9)...')
    rewards_ca2, opt_ca2 = run_bandit('constant_alpha', steps, runs, epsilon=epsilon, alpha=0.9, walk_std=walk_std)
    avg_rewards_sa = rewards_sa.mean(axis=0)
    avg_rewards_ca = rewards_ca.mean(axis=0)
    avg_rewards_ca2 = rewards_ca2.mean(axis=0)
    opt_actions_sa = opt_sa.mean(axis=0)
    opt_actions_ca = opt_ca.mean(axis=0)
    opt_actions_ca2 = opt_ca2.mean(axis=0)
    plot_results.extra_rewards = [avg_rewards_ca2]
    plot_results.extra_opt = [opt_actions_ca2]
    plot_results(avg_rewards_sa, opt_actions_sa, avg_rewards_ca, opt_actions_ca)

if __name__ == '__main__':
    main()
