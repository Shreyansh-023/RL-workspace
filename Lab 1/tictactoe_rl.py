"""
Tic-Tac-Toe Reinforcement Learning Agent
Using Temporal Difference Learning with Value Function
"""

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


class TicTacToe:
    """Tic-Tac-Toe game environment"""
    
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0: empty, 1: player X, -1: player O
        self.current_player = -1  # 1 for X, -1 for O
        
    def reset(self):
        """Reset the board to initial state"""
        self.board = np.zeros(9, dtype=int)
        self.current_player = -1
        
    def get_state(self):
        """Return current board state as tuple (hashable)"""
        return tuple(self.board)
    
    def legal_actions(self):
        """Return list of legal actions (empty cell indices)"""
        return np.where(self.board == 0)[0].tolist()
    
    def play_action(self, action):
        """
        Execute move and switch player
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
            (is_terminal, reward_for_player_1)
        """
        board = self.board.reshape(3, 3)
        
        # Check rows
        for row in board:
            if abs(sum(row)) == 3:
                return True, sum(row) / 3
        
        # Check columns
        for col in board.T:
            if abs(sum(col)) == 3:
                return True, sum(col) / 3
        
        # Check diagonals
        if abs(sum(np.diag(board))) == 3:
            return True, sum(np.diag(board)) / 3
        if abs(sum(np.diag(np.flipud(board)))) == 3:
            return True, sum(np.diag(np.flipud(board))) / 3
        
        # Check for draw
        if len(self.legal_actions()) == 0:
            return True, 0
        
        return False, None
    
    def outcome_for_agent(self, player_id):
        """
        Get reward from player's perspective
        Args:
            player_id: 1 for player X, -1 for player O
        Returns:
            Reward: +1 for win, -1 for loss, 0 for draw
        """
        is_term, reward = self.is_terminal()
        if not is_term:
            return None
        
        if reward == 0:
            return 0  # Draw
        elif reward == player_id:
            return 1  # Win
        else:
            return -1  # Loss


class RLAgent:
    """Reinforcement Learning Agent for Tic-Tac-Toe"""
    
    def __init__(self, player_id=-1, learning_rate=0.1, epsilon=0.1):
        """
        Initialize RL Agent
        Args:
            player_id: 1 for X, -1 for O
            learning_rate: alpha for TD update
            epsilon: exploration rate for epsilon-greedy
        """
        self.player_id = player_id
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.value_function = defaultdict(float)  # Initialize V(s) = 0 for all states
        self.move_history = []  # Track states for TD update
        
    def reset_episode(self):
        """Reset move history at start of episode"""
        self.move_history = []
    
    def choose_action(self, game, greedy=False):
        """
        Choose action using epsilon-greedy or greedy strategy
        Args:
            game: TicTacToe environment
            greedy: if True, use greedy; if False, use epsilon-greedy
        Returns:
            action (index 0-8)
        """
        legal_actions = game.legal_actions()
        
        if not greedy and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(legal_actions)
        
        # Exploitation: greedy action
        if not legal_actions:
            return None
        
        best_actions = []
        best_value = -float('inf')
        
        for action in legal_actions:
            # Simulate the action
            game.board[action] = self.player_id
            next_state = game.get_state()
            value = self.value_function[next_state]
            game.board[action] = 0  # Undo
            
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def update_values(self, game_states, reward):
        """
        Update value function using Temporal Difference learning
        V(s) <- V(s) + alpha * (V(s') - V(s))
        For terminal state: V(s) <- V(s) + alpha * (reward - V(s))
        
        Args:
            game_states: list of states visited during episode
            reward: final reward received
        """
        # Reverse to go from end to start
        for i in range(len(game_states) - 1, 0, -1):
            current_state = game_states[i - 1]
            next_state = game_states[i]
            
            # TD update
            next_value = self.value_function[next_state]
            self.value_function[current_state] += self.alpha * (next_value - self.value_function[current_state])
        
        # Update last state with final reward
        if game_states:
            last_state = game_states[-1]
            self.value_function[last_state] += self.alpha * (reward - self.value_function[last_state])


class TicTacToeTrainer:
    """Trainer for Tic-Tac-Toe RL agents"""
    
    def __init__(self, learning_rate=0.1, epsilon=0.1):
        self.agent_x = RLAgent(player_id=-1, learning_rate=learning_rate, epsilon=epsilon)
        self.agent_o = RLAgent(player_id=1, learning_rate=learning_rate, epsilon=epsilon)
        self.x_wins = []
        self.o_wins = []
        self.draws = []
    
    def play_game(self, agent_x_greedy=False, agent_o_greedy=False):
        """
        Play one game between two agents
        Args:
            agent_x_greedy: if True, agent X uses greedy; else epsilon-greedy
            agent_o_greedy: if True, agent O uses greedy; else epsilon-greedy
        Returns:
            reward_x, reward_o, winner (1 for X, -1 for O, 0 for draw)
        """
        game = TicTacToe()
        self.agent_x.reset_episode()
        self.agent_o.reset_episode()
        
        x_states = [game.get_state()]
        o_states = []
        
        while True:
            # Player X move
            action = self.agent_x.choose_action(game, greedy=agent_x_greedy)
            if action is None:
                break
            
            game.play_action(action)
            x_states.append(game.get_state())
            
            # Check terminal
            is_terminal, _ = game.is_terminal()
            if is_terminal:
                break
            
            # Player O move
            action = self.agent_o.choose_action(game, greedy=agent_o_greedy)
            if action is None:
                break
            
            game.play_action(action)
            o_states.append(game.get_state())
            
            # Check terminal
            is_terminal, _ = game.is_terminal()
            if is_terminal:
                break
        
        # Get outcomes
        reward_x = game.outcome_for_agent(1)
        reward_o = game.outcome_for_agent(-1)
        
        # Update value functions
        self.agent_x.update_values(x_states, reward_x)
        self.agent_o.update_values(o_states, reward_o)
        
        # Determine winner
        if reward_x == 1:
            winner = 1
        elif reward_o == 1:
            winner = -1
        else:
            winner = 0
        
        return reward_x, reward_o, winner
    
    def train(self, episodes=1000, agent_x_greedy=False, agent_o_greedy=False):
        """
        Training loop
        Args:
            episodes: number of games to play
            agent_x_greedy: agent X strategy
            agent_o_greedy: agent O strategy
        """
        x_wins_count = 0
        o_wins_count = 0
        draws_count = 0
        self.x_win_probs = []
        self.x_draw_probs = []
        self.x_lose_probs = []
        self.experiment_ids = []
        for episode in range(episodes):
            reward_x, reward_o, winner = self.play_game(agent_x_greedy, agent_o_greedy)
            if winner == 1:
                x_wins_count += 1
            elif winner == -1:
                o_wins_count += 1
            else:
                draws_count += 1
            # Record every 10 episodes
            if (episode + 1) % 10 == 0:
                total = x_wins_count + o_wins_count + draws_count
                self.x_win_probs.append(x_wins_count / total if total > 0 else 0)
                self.x_draw_probs.append(draws_count / total if total > 0 else 0)
                self.x_lose_probs.append(o_wins_count / total if total > 0 else 0)
                self.experiment_ids.append(episode + 1)
        print(f"\nTraining completed ({episodes} episodes)")
        print(f"Agent X Wins: {x_wins_count}")
        print(f"Agent O Wins: {o_wins_count}")
        print(f"Draws: {draws_count}")
    
    def plot_learning_curve(self, title=""):
        """Plot probability of win/draw/lose for Agent X vs experiments"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.experiment_ids, self.x_win_probs, label='X Win Probability', marker='o')
        plt.plot(self.experiment_ids, self.x_draw_probs, label='X Draw Probability', marker='^')
        plt.plot(self.experiment_ids, self.x_lose_probs, label='X Lose Probability', marker='s')
        plt.xlabel('Experiment (Episodes)')
        plt.ylabel('Probability')
        plt.title(f'Probability of Win/Draw/Lose for X - {title}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'probability_curve_{title.replace(" ", "_")}.png')
        plt.show()


def plot_all_scenarios():
    """Plot all 4 scenarios in a single 2x2 grid and save as one PNG."""
    scenarios = [
        (False, False, 'ε-Greedy vs ε-Greedy'),
        (True, False, 'Greedy vs ε-Greedy'),
        (False, True, 'ε-Greedy vs Greedy'),
        (True, True, 'Greedy vs Greedy'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (x_greedy, o_greedy, title) in enumerate(scenarios):
        trainer = TicTacToeTrainer(learning_rate=0.1, epsilon=0.1)
        trainer.train(episodes=10000, agent_x_greedy=x_greedy, agent_o_greedy=o_greedy)
        ax = axes[idx // 2, idx % 2]
        ax.plot(trainer.experiment_ids, trainer.x_win_probs, label='X Wins', color='blue')
        ax.plot(trainer.experiment_ids, trainer.x_lose_probs, label='O Wins', color='orange')
        ax.plot(trainer.experiment_ids, trainer.x_draw_probs, label='Draws', color='green')
        ax.set_title(f'Learning Curves: {title}')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_scenarios_learning_curves.png')
    plt.show()


def main():
    """Main function to run experiments"""
    print("="*60)
    print("TIC-TAC-TOE REINFORCEMENT LEARNING AGENT")
    print("="*60)
    plot_all_scenarios()
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
