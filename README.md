# Flappy Bird Q-Learning

A Flappy Bird AI agent based on the Q-Learning algorithm.

## File Overview

- `train.py`: Training script for training the AI in headless mode.
- `play.py`: Execution script for visualizing and evaluating the AI's gameplay.
- `flappy_q/agent.py`: Core Q-learning algorithm (states, actions, and reward updates).
- `flappy_q/env.py`: Game environment (physics engine and collision detection).
- `flappy_q/ui_pygame.py`: Game UI (Pygame rendering and text display).
- `flappy_q/io_paths.py`: File path management (loading models, images, and audio assets).
- `data/qvalues.json`: Saved Q-table containing the AI's learned knowledge.
- `data/hitmasks_data.pkl`: Pre-calculated pixel-perfect collision masks to speed up training.

## How to Run
Open your terminal (or command prompt) and run the following commands:

1. Install the required library:
pip install pygame
2. Initialize game data (generate collision masks, run only once):
python play.py --dump_hitmasks
3. Train the AI agent:
python train.py --iter 10000 (Add --verbose to print training logs)
(You can change 10000 to any desired number of episodes. Can be safely interrupted and saved at any time with Ctrl+C)
4. Watch the AI agent play the game:
python play.py
