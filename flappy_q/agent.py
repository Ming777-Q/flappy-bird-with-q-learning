from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

Obs = str


class QAgent:
    def __init__(self):
        # Q-table: A dictionary mapping state strings to a list of Q-values [Q(s, action=0), Q(s, action=1)]
        self.qvalues = {}

        # RL Hyperparameters:
        # alpha: Learning rate. Determines to what extent newly acquired information overrides old information.
        self.alpha = 0.7
        # gamma: Discount factor. Determines the importance of future rewards. 1.0 means long-term rewards are prioritized.
        self.gamma = 1.0
        # epsilon: Exploration rate. Controls the Exploration vs. Exploitation trade-off.
        self.epsilon = 0.1
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.00001

        # Memory buffer to store the trajectory (s, a, s', r) of the current episode
        self.moves: List[Tuple[Obs, int, Obs, float]] = []

    def _init_state_if_null(self, state: Obs) -> None:
        # Lazily initialize Q-values for unseen states to [0.0, 0.0]
        if state not in self.qvalues:
            self.qvalues[state] = [0.0, 0.0]

    def choose_action(self, obs: Obs, train: bool = True) -> int:
        """
        [Core Module: Action Selection]
        Implements the epsilon-greedy strategy to balance Exploration and Exploitation.
        """
        self._init_state_if_null(obs)

        # Exploration: With probability epsilon, choose a random action to explore the environment
        if train and random.random() <= self.epsilon:
            return random.choice([0, 1])

        # Exploitation: With probability (1 - epsilon), choose the action with the highest Q-value
        if self.qvalues[obs][0] >= self.qvalues[obs][1]:
            return 0  # Do nothing
        else:
            return 1  # Flap

    def remember(self, obs: Obs, action: int, next_obs: Obs, reward: float) -> None:
        # Store the transition tuple for offline updates at the end of the episode
        self.moves.append((obs, action, next_obs, reward))

    def update_from_history(self) -> None:
        """
        [Core Module: Reward Shaping & Bellman Update]
        Updates the Q-table by iterating backwards through the stored episode trajectory.
        This offline update helps propagate the delayed penalty of crashing back to the actions that caused it.
        """
        if not self.moves:
            return

        # Reverse the history to perform a backward pass
        history = list(reversed(self.moves))
        last_state = history[0][0]

        # Heuristic check: Did the bird die by hitting the ceiling? (y-coordinate check)
        try:
            last_y0 = int(last_state.split("_")[1])
            high_death_flag = True if last_y0 > 120 else False
        except:
            high_death_flag = False

        t = 0
        last_flap = True

        for move in history:
            t += 1
            obs, action, next_obs, _ = move

            self._init_state_if_null(obs)
            self._init_state_if_null(next_obs)

            # [Reward Shaping]: Assigning heavy penalties for actions directly leading to a crash.
            # Why? The base environment gives +1 for survival, but the agent only realizes it failed at the very end.
            # We override the base reward with -1000 for the last 2 steps or the fatal flap to strongly discourage suicidal actions.
            if t <= 2:
                cur_reward = -1000
                if action == 1:
                    last_flap = False
            elif (last_flap or high_death_flag) and action == 1:
                cur_reward = -1000
                high_death_flag = False
                last_flap = False
            else:
                # If it's a safe historical state, use a neutral/base reward
                cur_reward = 0

            # [Bellman Update]: The core Q-Learning equation
            # Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * (R + gamma * max Q(s', a'))
            max_next_q = max(self.qvalues[next_obs][0], self.qvalues[next_obs][1])
            current_q = self.qvalues[obs][action]

            new_q = (1 - self.alpha) * current_q + self.alpha * (cur_reward + self.gamma * max_next_q)
            self.qvalues[obs][action] = new_q

        # Clear the memory for the next episode
        self.moves = []

        # Decay epsilon over time to gradually shift from exploration to pure exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def save_q(self, path: Path) -> None:
        file_path = str(path).replace(".npy", ".json")
        with open(file_path, "w") as f:
            json.dump(self.qvalues, f)

    def load_q(self, path: Path) -> None:
        file_path = str(path).replace(".npy", ".json")
        if Path(file_path).exists():
            with open(file_path, "r") as f:
                self.qvalues = json.load(f)
        else:
            print(f"[{file_path}] does not exist. Initializing with an empty dictionary (expected for a fresh start).")