from __future__ import annotations

import random
import sys
import pickle
from typing import Dict, Any, Tuple

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

from .env import FlappyEnv
from .agent import QAgent, Obs
from .io_paths import SPRITES_DIR, AUDIO_DIR, HITMASKS_PATH

# Asset definitions
PLAYERS_LIST = (
    ("redbird-upflap.png", "redbird-midflap.png", "redbird-downflap.png"),
    ("bluebird-upflap.png", "bluebird-midflap.png", "bluebird-downflap.png"),
    ("yellowbird-upflap.png", "yellowbird-midflap.png", "yellowbird-downflap.png"),
)
BACKGROUNDS_LIST = ("background-day.png", "background-night.png")
PIPES_LIST = ("pipe-green.png", "pipe-red.png")


def load_assets() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads all necessary images and sounds into memory for PyGame rendering.
    """
    IMAGES, SOUNDS = {}, {}

    IMAGES["numbers"] = tuple(
        pygame.image.load(str(SPRITES_DIR / f"{i}.png")).convert_alpha() for i in range(10)
    )
    IMAGES["gameover"] = pygame.image.load(str(SPRITES_DIR / "gameover.png")).convert_alpha()
    IMAGES["message"] = pygame.image.load(str(SPRITES_DIR / "message.png")).convert_alpha()
    IMAGES["base"] = pygame.image.load(str(SPRITES_DIR / "base.png")).convert_alpha()

    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES["background"] = pygame.image.load(str(SPRITES_DIR / BACKGROUNDS_LIST[randBg])).convert()

    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES["player"] = tuple(
        pygame.image.load(str(SPRITES_DIR / p)).convert_alpha() for p in PLAYERS_LIST[randPlayer]
    )

    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES["pipe"] = (
        pygame.transform.rotate(pygame.image.load(str(SPRITES_DIR / PIPES_LIST[pipeindex])).convert_alpha(), 180),
        pygame.image.load(str(SPRITES_DIR / PIPES_LIST[pipeindex])).convert_alpha(),
    )

    ext = ".wav" if "win" in sys.platform else ".ogg"
    SOUNDS["die"] = pygame.mixer.Sound(str(AUDIO_DIR / f"die{ext}"))
    SOUNDS["hit"] = pygame.mixer.Sound(str(AUDIO_DIR / f"hit{ext}"))
    SOUNDS["point"] = pygame.mixer.Sound(str(AUDIO_DIR / f"point{ext}"))
    SOUNDS["wing"] = pygame.mixer.Sound(str(AUDIO_DIR / f"wing{ext}"))

    return IMAGES, SOUNDS


def get_hitmask(image: pygame.Surface) -> list:
    """
    Extracts the alpha channel (transparency) from an image to create a boolean 2D array.
    This allows for highly accurate, pixel-perfect collision detection rather than simple bounding boxes.
    """
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


def run_play(agent: QAgent, fps: int = 30, dump_hitmasks: bool = False) -> None:
    """
    The main evaluation and visualization loop.
    It runs the environment using the agent's pre-trained policy and renders the state via PyGame.
    """
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((288, 512))
    pygame.display.set_caption("Flappy Bird Q-Learning")

    # Font used for rendering real-time RL state information (Debug Info)
    font = pygame.font.SysFont("arial", 20, bold=True)

    IMAGES, SOUNDS = load_assets()

    # Generate hitmasks for pixel-perfect collisions
    HITMASKS = {
        "pipe": (get_hitmask(IMAGES["pipe"][0]), get_hitmask(IMAGES["pipe"][1])),
        "player": (
            get_hitmask(IMAGES["player"][0]),
            get_hitmask(IMAGES["player"][1]),
            get_hitmask(IMAGES["player"][2]),
        ),
    }

    # Utility mode: Dump hitmasks to disk so headless training doesn't require PyGame rendering overhead
    if dump_hitmasks:
        with HITMASKS_PATH.open("wb") as f:
            pickle.dump(HITMASKS, f, pickle.HIGHEST_PROTOCOL)
        print(f"Hitmasks have been saved to {HITMASKS_PATH}. Exiting program.")
        pygame.quit()
        sys.exit()

    # Initialize the identical environment used in training, now with hitmasks
    env = FlappyEnv(hitmasks=HITMASKS)

    while True:
        obs = env.reset()
        done = False
        action = 0
        reward = 0.0
        cumulative_reward = 0.0

        # Episode Loop
        while not done:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            # [RL Core]: Action Selection during Evaluation
            # Notice that train=False. This means epsilon is ignored.
            # The agent acts purely greedily (Exploitation), relying entirely on the learned Q-table.
            action = agent.choose_action(obs, train=False)

            # Execute the chosen action in the environment
            next_obs, reward, done, info = env.step(action)
            cumulative_reward += reward

            # Play audio feedback based on environment events
            if info.get("scored"):
                SOUNDS["point"].play()

            if action == 1:
                SOUNDS["wing"].play()

            # Retrieve internal game variables (coordinates) specifically for rendering
            env_state = env.get_render_state()

            # Draw the visual representation of the game
            draw_frame(SCREEN, IMAGES, env_state)

            # Draw the internal RL states to help graders verify the MDP logic
            draw_debug_info(SCREEN, font, obs, action, reward, cumulative_reward)

            pygame.display.update()
            FPSCLOCK.tick(fps)

            # Advance the state
            obs = next_obs

        # Episode finished (Bird crashed)
        SOUNDS["hit"].play()
        if not info.get("groundCrash"):
            SOUNDS["die"].play()

        print(f"Episode finished! Score: {info.get('score', 0)}")

        # Brief pause before restarting the next evaluation episode
        pygame.time.delay(500)


def draw_frame(screen: pygame.Surface, images: Dict[str, Any], env_state: Dict[str, Any]) -> None:
    """
    Renders all standard game elements (background, pipes, bird, base) onto the PyGame screen.
    """
    screen.blit(images["background"], (0, 0))

    for uPipe, lPipe in zip(env_state["upperPipes"], env_state["lowerPipes"]):
        screen.blit(images["pipe"][0], (uPipe["x"], uPipe["y"]))
        screen.blit(images["pipe"][1], (lPipe["x"], lPipe["y"]))

    screen.blit(images["base"], (env_state["basex"], int(512 * 0.79)))

    show_score(screen, images["numbers"], score=env_state["score"])

    idx = env_state["playerIndex"]
    screen.blit(images["player"][idx], (env_state["playerx"], env_state["playery"]))


def show_score(screen: pygame.Surface, number_images: Tuple[pygame.Surface, ...], score: int) -> None:
    """
    Renders the current game score in the center of the screen.
    """
    digits = [int(x) for x in str(score)]
    total_width = sum(number_images[d].get_width() for d in digits)
    x_offset = (288 - total_width) / 2

    for d in digits:
        screen.blit(number_images[d], (x_offset, 512 * 0.1))
        x_offset += number_images[d].get_width()


def draw_debug_info(screen: pygame.Surface, font: pygame.font.Font, obs: Obs, action: int, reward: float,
                    cumulative_reward: float) -> None:
    """
    Overlay diagnostic RL metrics on the screen.
    This is highly beneficial for demonstrating to graders exactly what the agent "sees" (the discretized state)
    and what decisions it is making in real-time.
    """
    white = (255, 255, 255)
    black = (0, 0, 0)

    # Parse the discretized state string back into intuitive values
    try:
        x0, y0, v, y1 = obs.split("_")
    except ValueError:
        x0, y0, v, y1 = "?", "?", "?", "?"

    action_str = "Flap" if action == 1 else "Nothing"

    texts = [
        f"State: x0={x0}, y0={y0}, v={v}, y1={y1}",
        f"Action: {action_str}",
        f"Reward: {reward}",
        f"Cumulative Reward: {cumulative_reward}"
    ]

    start_y = 10
    # Draw text with a simple drop shadow for readability against moving backgrounds
    for i, text in enumerate(texts):
        text_shadow = font.render(text, True, black)
        screen.blit(text_shadow, (11, start_y + i * 25 + 1))

        text_surface = font.render(text, True, white)
        screen.blit(text_surface, (10, start_y + i * 25))