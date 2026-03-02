from __future__ import annotations

import argparse

from flappy_q.agent import QAgent
from flappy_q.io_paths import QVALUES_PATH
from flappy_q.ui_pygame import run_play


def main() -> None:
    # Set up command-line arguments for the evaluation script
    parser = argparse.ArgumentParser("play.py")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the PyGame display.")
    parser.add_argument("--dump_hitmasks", action="store_true", help="Generate and save pixel-perfect collision masks.")
    args = parser.parse_args()

    # Initialize the Q-Learning agent for evaluation
    agent = QAgent()

    # Load the pre-trained Q-table from the disk.
    # This allows the agent to utilize the knowledge acquired during headless training.
    agent.load_q(QVALUES_PATH)

    print("Initializing visualization mode...")

    # Warn the user/grader if they are trying to evaluate an untrained agent
    if not QVALUES_PATH.exists():
        print("Warning: Pre-trained Q-table not found. The agent is untrained. Please run train.py first!")

    # Start the PyGame visualization loop.
    # The agent will now play the game using its learned policy.
    run_play(agent, fps=args.fps, dump_hitmasks=args.dump_hitmasks)


if __name__ == "__main__":
    main()