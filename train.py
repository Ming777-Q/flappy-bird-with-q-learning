from __future__ import annotations

import argparse
import pickle
from collections import deque

from flappy_q.agent import QAgent
from flappy_q.env import FlappyEnv
# Cross-file reference: Loading standardized data paths defined in io_paths.py
from flappy_q.io_paths import QVALUES_PATH, HITMASKS_PATH


def main() -> None:
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--iter", type=int, default=10000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    iterations = args.iter
    verbose = args.verbose

    # Load pre-generated hitmasks (pixel-perfect collision boxes)
    try:
        with HITMASKS_PATH.open("rb") as f:
            hitmasks = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find {HITMASKS_PATH}.\n"
            "Please run the GUI program once to generate the hitmasks. Example:\n"
            "  python play.py --dump_hitmasks\n"
        )

    # Initialize the Environment and the RL Agent
    env = FlappyEnv(hitmasks=hitmasks)
    agent = QAgent()

    # Load existing Q-table if it exists (allows resuming training)
    agent.load_q(QVALUES_PATH)

    print(f"Starting headless training for {iterations} episodes...")
    print(
        "Note: You can press Ctrl+C at any time to interrupt training. The current Q-table will be saved automatically.")

    max_score_all = 0
    recent_scores = deque(maxlen=100)  # Track moving average for performance monitoring

    try:
        # Standard Reinforcement Learning Loop
        for episode in range(iterations):
            # 1. Reset environment to get initial state observation
            obs = env.reset()
            done = False
            score = 0

            # 2. Step through the environment until episode terminates (game over)
            while not done:
                # 3. Agent selects an action based on the current state (epsilon-greedy)
                action = agent.choose_action(obs, train=True)

                # 4. Environment processes the action and returns the next state and base reward
                next_obs, reward, done, info = env.step(action)

                # 5. Store the transition (s, a, s', r) in the agent's memory
                agent.remember(obs, action, next_obs, reward)

                # 6. Advance the state
                obs = next_obs
                score = info.get("score", 0)

                # Early stopping mechanism if the agent performs extremely well
                if score > 999:
                    done = True

            # 7. Offline Q-value Update
            # At the end of the episode, apply the Bellman equation across the saved trajectory
            agent.update_from_history()

            recent_scores.append(score)
            avg_score = sum(recent_scores) / len(recent_scores)

            if score > max_score_all:
                max_score_all = score

            if verbose:
                print(
                    f"Episode: {episode + 1:5d} | Score: {score:5d} | Avg Score: {avg_score:6.2f} | Max Score: {max_score_all:5d} | Epsilon: {agent.epsilon:.4f}")
            elif (episode + 1) % 100 == 0:
                print(
                    f"Episode: {episode + 1:5d} | Avg Score: {avg_score:6.2f} | Max Score: {max_score_all:5d} | Epsilon: {agent.epsilon:.4f}")

    except KeyboardInterrupt:
        print("\n[Warning] Manual interruption (Ctrl+C) detected! Saving Q-table...")

    finally:
        # Persist the trained Q-table to disk for future evaluation via play.py
        print(f"Saving Q-table to {QVALUES_PATH}...")
        agent.save_q(QVALUES_PATH)
        print("Save successful! You can now run play.py to evaluate the training results.")


if __name__ == "__main__":
    main()
