"""
Tic-Tac-Toe Reinforcement Learning Agent following the provided flowchart.
Outputs learning curves similar to the reference plots: average reward and
percentage of optimal actions over episodes for different epsilon values.
"""

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


# All eight winning line indices for quick checks
WIN_PATTERNS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def board_outcome(flat_board):
    """Return 1 if X wins, -1 if O wins, 0 if draw, None if game ongoing."""
    for a, b, c in WIN_PATTERNS:
        s = flat_board[a] + flat_board[b] + flat_board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    if 0 not in flat_board:
        return 0  # draw
    return None  # not terminal


def best_action_by_value(game, player_id, value_function):
    """Greedy action according to value function (ties broken randomly)."""
    legal_actions = game.legal_actions()
    if not legal_actions:
        return None

    best_actions = []
    best_value = -float("inf")

    for action in legal_actions:
        game.board[action] = player_id
        next_state = game.get_state()
        value = value_function[next_state]
        game.board[action] = 0

        if value > best_value:
            best_value = value
            best_actions = [action]
        elif value == best_value:
            best_actions.append(action)

    return random.choice(best_actions)


def winning_actions(flat_board, player_id):
    """Return all legal actions that immediately win for player_id."""
    wins = []
    for idx, cell in enumerate(flat_board):
        if cell != 0:
            continue
        temp = flat_board.copy()
        temp[idx] = player_id
        if board_outcome(temp) == player_id:
            wins.append(idx)
    return wins


def is_optimal_action(flat_board, player_id, action):
    """Heuristic optimal action used for the plotted metric.

    Priority: winning move > blocking move > center > corner > edge.
    """
    if flat_board[action] != 0:
        return False

    win_now = winning_actions(flat_board, player_id)
    if win_now and action in win_now:
        return True

    block_now = winning_actions(flat_board, -player_id)
    if block_now and action in block_now:
        return True

    if action == 4:  # center
        return True

    if action in (0, 2, 6, 8):  # corners
        return True

    return False  # edges are lowest priority


class TicTacToe:
    """Tic-Tac-Toe game environment"""

    def __init__(self, start_player=1):
        self.board = np.zeros(9, dtype=int)  # 0: empty, 1: player X, -1: player O
        self.current_player = start_player  # who starts can be biased/alternating

    def reset(self, start_player=1):
        """Reset the board to initial state"""
        self.board = np.zeros(9, dtype=int)
        self.current_player = start_player

    def get_state(self):
        """Return current board state as tuple (hashable)"""
        return tuple(self.board)

    def legal_actions(self):
        """Return list of legal actions (empty cell indices)"""
        return np.where(self.board == 0)[0].tolist()

    def play_action(self, action):
        """
        Execute move for the current player and switch turn.
        Args:
            action: index of cell (0-8)
        Returns:
            True if move is valid, False otherwise
        """
        if self.board[action] != 0:
            return False
        self.board[action] = self.current_player
        self.current_player *= -1  # Switch player
        return True

    def is_terminal(self):
        """
        Check if game is terminal (win, loss, or draw)
        Returns:
            (is_terminal, reward_for_player_X)
        """
        outcome = board_outcome(self.board)
        if outcome is None:
            return False, None
        return True, outcome

    def outcome_for_agent(self, player_id):
        """Reward from player's perspective.

        Reward shaping: win=1.0, draw=0.5, loss=0.0.
        """
        is_term, reward = self.is_terminal()
        if not is_term:
            return None
        if reward == player_id:
            return 1.0
        if reward == 0:
            return 0.5
        return 0.0


class RLAgent:
    """Reinforcement Learning Agent for Tic-Tac-Toe"""

    def __init__(self, player_id=1, learning_rate=0.1, epsilon=0.1):
        self.player_id = player_id
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.value_function = defaultdict(float)  # V(s)
        self.move_history = []  # Track states for TD updates

    def reset_episode(self):
        self.move_history = []

    def choose_action(self, game, greedy=False):
        """Epsilon-greedy or greedy action selection."""
        legal_actions = game.legal_actions()

        if not legal_actions:
            return None

        explore = (not greedy) and (random.random() < self.epsilon)
        if explore:
            return random.choice(legal_actions)

        best_actions = []
        best_value = -float("inf")

        for action in legal_actions:
            game.board[action] = self.player_id  # simulate
            next_state = game.get_state()
            value = self.value_function[next_state]
            game.board[action] = 0  # undo

            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def update_values(self, game_states, reward):
        """Temporal Difference update along the recorded trajectory."""
        for i in range(len(game_states) - 1, 0, -1):
            current_state = game_states[i - 1]
            next_state = game_states[i]
            next_value = self.value_function[next_state]
            self.value_function[current_state] += self.alpha * (
                next_value - self.value_function[current_state]
            )

        if game_states:
            last_state = game_states[-1]
            self.value_function[last_state] += self.alpha * (
                reward - self.value_function[last_state]
            )


class TicTacToeTrainer:
    """Trainer for Tic-Tac-Toe RL agents"""

    def __init__(self, learning_rate=0.1, epsilon_x=0.1, epsilon_o=0.1):
        self.agent_x = RLAgent(player_id=1, learning_rate=learning_rate, epsilon=epsilon_x)
        self.agent_o = RLAgent(player_id=-1, learning_rate=learning_rate, epsilon=epsilon_o)

    def _pick_start_player(self, mode, episode_idx):
        if mode == "alternate":
            return 1 if episode_idx % 2 == 0 else -1
        if mode == "random":
            return random.choice([1, -1])
        if mode == "fixed_o":
            return -1
        # default fixed X starts
        return 1

    def play_game(self, agent_x_greedy=False, agent_o_greedy=False, track_optimal=False, start_player=1):
        """
        Play one self-play game.
        Returns reward_x, reward_o, winner, optimal_moves, total_x_moves when tracking.
        """
        game = TicTacToe(start_player=start_player)
        self.agent_x.reset_episode()
        self.agent_o.reset_episode()

        x_states = []
        o_states = []

        # seed trajectory list with the initial state for whichever player moves first
        if game.current_player == 1:
            x_states.append(game.get_state())
        else:
            o_states.append(game.get_state())

        optimal_moves = 0
        total_x_moves = 0

        while True:
            current_player = game.current_player
            if current_player == 1:
                agent = self.agent_x
                greedy_flag = agent_x_greedy
            else:
                agent = self.agent_o
                greedy_flag = agent_o_greedy

            action = agent.choose_action(game, greedy=greedy_flag)
            if action is None:
                break

            if track_optimal and current_player == 1:
                total_x_moves += 1
                greedy_action = best_action_by_value(
                    game, self.agent_x.player_id, self.agent_x.value_function
                )
                if greedy_action is not None and action == greedy_action:
                    optimal_moves += 1

            game.play_action(action)
            next_state = game.get_state()
            if current_player == 1:
                x_states.append(next_state)
            else:
                o_states.append(next_state)

            is_terminal, outcome = game.is_terminal()
            if is_terminal:
                break

        reward_x = game.outcome_for_agent(self.agent_x.player_id)
        reward_o = game.outcome_for_agent(self.agent_o.player_id)

        self.agent_x.update_values(x_states, reward_x)
        self.agent_o.update_values(o_states, reward_o)

        # Determine winner using outcome sign (1 X, -1 O, 0 draw)
        _, outcome = game.is_terminal()
        winner = outcome if outcome is not None else 0

        return reward_x, reward_o, winner, optimal_moves, total_x_moves

    def train_with_metrics(self, episodes=1000, agent_x_greedy=False, agent_o_greedy=False, smooth_window=50, start_mode="fixed"):
        """Run training and return smoothed averages for reward and optimal moves.

        start_mode: "fixed" (always X), "alternate", or "random".
        """
        rewards = []
        optimal_rates = []

        cumulative_optimal = 0
        cumulative_moves = 0

        for ep in range(episodes):
            start_player = self._pick_start_player(start_mode, ep)
            reward_x, _, _, opt_moves, x_moves = self.play_game(
                agent_x_greedy=agent_x_greedy,
                agent_o_greedy=agent_o_greedy,
                track_optimal=True,
                start_player=start_player,
            )

            rewards.append(reward_x)
            cumulative_optimal += opt_moves
            cumulative_moves += x_moves
            if cumulative_moves == 0:
                optimal_rates.append(0.0)
            else:
                optimal_rates.append(cumulative_optimal / cumulative_moves)

        # Smooth with simple moving average
        def moving_average(seq, window):
            window = max(1, window)
            cumsum = np.cumsum(np.insert(seq, 0, 0))
            vals = (cumsum[window:] - cumsum[:-window]) / window
            # Pad the start to keep length equal
            pad = [vals[0]] * (window - 1)
            return pad + vals.tolist()

        smoothed_reward = moving_average(rewards, smooth_window)
        smoothed_optimal = moving_average(optimal_rates, smooth_window)
        return smoothed_reward, smoothed_optimal


def plot_learning_curves(episodes, epsilon_values, rewards_dict, optimal_dict, title_suffix="", filename="tictactoe_learning_curves.png"):
    steps = np.arange(1, episodes + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for eps in epsilon_values:
        axes[0].plot(steps, rewards_dict[eps], label=f"ε = {eps}")
    axes[0].set_ylabel("Average reward")
    axes[0].set_ylim(0.0, 1.6)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for eps in epsilon_values:
        axes[1].plot(steps, np.array(optimal_dict[eps]) * 100.0, label=f"ε = {eps}")
    axes[1].set_xlabel("Steps (episodes)")
    axes[1].set_ylabel("Optimal action (%)")
    axes[1].set_ylim(30, 100)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Epsilon-Greedy TD Learning on Tic-Tac-Toe {title_suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filename, dpi=150)
    plt.show()


def main():
    """Run experiments for multiple epsilon values and plot learning curves."""

    print("=" * 60)
    print("TIC-TAC-TOE TD LEARNING (GREEDY VS EPSILON-GREEDY)")
    print("=" * 60)

    episodes = 3000
    epsilon_values = [0.0, 0.01, 0.1]

    # Case 1: X always starts (bias toward X)
    rewards_x_start = {}
    optimal_x_start = {}
    for eps in epsilon_values:
        print(f"\n[X-start] Agent X ε={eps} vs greedy O, fixed X start")
        trainer = TicTacToeTrainer(learning_rate=0.1, epsilon_x=eps, epsilon_o=0.0)
        rewards, optimal = trainer.train_with_metrics(
            episodes=episodes,
            agent_x_greedy=False,
            agent_o_greedy=True,
            smooth_window=50,
            start_mode="fixed_x",
        )
        rewards_x_start[eps] = rewards
        optimal_x_start[eps] = optimal

    plot_learning_curves(
        episodes,
        epsilon_values,
        rewards_x_start,
        optimal_x_start,
        title_suffix="(X starts)",
        filename="tictactoe_learning_curves_x_start.png",
    )

    # Case 2: O always starts (bias toward O)
    rewards_o_start = {}
    optimal_o_start = {}
    for eps in epsilon_values:
        print(f"\n[O-start] Agent X ε={eps} vs greedy O, fixed O start")
        trainer = TicTacToeTrainer(learning_rate=0.1, epsilon_x=eps, epsilon_o=0.0)
        rewards, optimal = trainer.train_with_metrics(
            episodes=episodes,
            agent_x_greedy=False,
            agent_o_greedy=True,
            smooth_window=50,
            start_mode="fixed_o",
        )
        rewards_o_start[eps] = rewards
        optimal_o_start[eps] = optimal

    plot_learning_curves(
        episodes,
        epsilon_values,
        rewards_o_start,
        optimal_o_start,
        title_suffix="(O starts)",
        filename="tictactoe_learning_curves_o_start.png",
    )

    print("\nSaved plots to tictactoe_learning_curves_x_start.png and tictactoe_learning_curves_o_start.png")
    print("Training complete.")


if __name__ == "__main__":
    main()
