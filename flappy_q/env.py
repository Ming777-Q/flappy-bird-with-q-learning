from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
import random
from typing import Dict, List, Optional, Tuple, Any

import pygame

Obs = str


@dataclass
class EnvConfig:
    screen_width: int = 288
    screen_height: int = 512
    pipe_gap_size: int = 100
    base_y_ratio: float = 0.79
    pipe_w: int = 52
    pipe_h: int = 320
    player_w: int = 34
    player_h: int = 24
    base_w: int = 336
    background_w: int = 288
    pipe_vel_x: int = -4
    player_vel_y_init: int = -9
    player_max_vel_y: int = 10
    player_min_vel_y: int = -8
    player_acc_y: int = 1
    player_flap_acc: int = -9


class FlappyEnv:
    def __init__(self, config: Optional[EnvConfig] = None, hitmasks: Optional[Dict[str, Any]] = None):
        self.cfg = config or EnvConfig()
        self.HITMASKS = hitmasks
        self.BASEY = self.cfg.screen_height * self.cfg.base_y_ratio
        self.base_shift = self.cfg.base_w - self.cfg.background_w

        self.playerIndexGen = None
        self.playerIndex = 0
        self.loopIter = 0
        self.playerx = 0
        self.playery = 0
        self.basex = 0
        self.upperPipes: List[Dict[str, float]] = []
        self.lowerPipes: List[Dict[str, float]] = []
        self.score = 0
        self.playerVelY = self.cfg.player_vel_y_init
        self.playerFlapped = False

    def reset(self) -> Obs:
        self.score = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])

        self.playerx = int(self.cfg.screen_width * 0.2)
        self.playery = int((self.cfg.screen_height - self.cfg.player_h) / 2)
        self.basex = 0

        newPipe1 = self._get_random_pipe()
        newPipe2 = self._get_random_pipe()

        self.upperPipes = [
            {"x": self.cfg.screen_width + 200, "y": newPipe1[0]["y"]},
            {"x": self.cfg.screen_width + 200 + (self.cfg.screen_width / 2), "y": newPipe2[0]["y"]},
        ]
        self.lowerPipes = [
            {"x": self.cfg.screen_width + 200, "y": newPipe1[1]["y"]},
            {"x": self.cfg.screen_width + 200 + (self.cfg.screen_width / 2), "y": newPipe2[1]["y"]},
        ]

        self.playerVelY = self.cfg.player_vel_y_init
        self.playerFlapped = False

        return self._get_observation()

    def step(self, action: int) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment based on the agent's action.
        Returns: (next_observation, reward, done_flag, info_dict)
        """
        # Action 1 means 'flap'. Apply upward acceleration.
        if action == 1:
            if self.playery > -2 * self.cfg.player_h:
                self.playerVelY = self.cfg.player_flap_acc
                self.playerFlapped = True

        # Check for collisions with pipes or the ground
        crashed, groundCrash = self._check_crash(
            player={"x": self.playerx, "y": self.playery, "index": self.playerIndex},
            upperPipes=self.upperPipes,
            lowerPipes=self.lowerPipes,
        )

        if crashed:
            info = self._snapshot_info(groundCrash=groundCrash)
            # Episode terminates. Return observation and 0 base reward (handled later via reward shaping in agent.py)
            return self._get_observation(), 0.0, True, info

        # Base Survival Reward
        reward = 1.0
        scored = False
        playerMidPos = self.playerx + self.cfg.player_w / 2

        # Check if the bird successfully passed a pipe
        for pipe in self.upperPipes:
            pipeMidPos = pipe["x"] + self.cfg.pipe_w / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                scored = True
                # Bonus reward for passing a pipe
                reward = 5.0

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.base_shift)

        if self.playerVelY < self.cfg.player_max_vel_y and not self.playerFlapped:
            self.playerVelY += self.cfg.player_acc_y
        if self.playerFlapped:
            self.playerFlapped = False

        playerHeight = self.cfg.player_h
        self.playery += min(self.playerVelY, self.BASEY - self.playery - playerHeight)

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe["x"] += self.cfg.pipe_vel_x
            lPipe["x"] += self.cfg.pipe_vel_x

        if 0 < self.upperPipes[0]["x"] < 5:
            newPipe = self._get_random_pipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        if self.upperPipes[0]["x"] < -self.cfg.pipe_w:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        next_obs = self._get_observation()
        info = self._snapshot_info(groundCrash=False)
        info["scored"] = scored

        return next_obs, reward, False, info

    def render(self) -> None:
        return

    def get_render_state(self) -> Dict[str, Any]:
        return {
            "playerx": self.playerx,
            "playery": self.playery,
            "playerIndex": self.playerIndex,
            "basex": self.basex,
            "upperPipes": self.upperPipes,
            "lowerPipes": self.lowerPipes,
            "score": self.score,
        }

    def _get_observation(self) -> Obs:
        """
        [Core Module: State Extraction & Discretization]
        Converts the continuous pixel space of the game into a simplified, discrete state string.
        Why? Tabular Q-learning cannot handle infinite continuous states. We must bucketize the coordinates
        so the Q-table remains small enough to converge efficiently.
        """
        pipe0 = self.lowerPipes[0]
        pipe1 = self.lowerPipes[1] if len(self.lowerPipes) > 1 else self.lowerPipes[0]

        # Determine which pipe is the immediate next obstacle
        if self.playerx - pipe0["x"] >= 50:
            pipe0 = self.lowerPipes[1]
            if len(self.lowerPipes) > 2:
                pipe1 = self.lowerPipes[2]

        # Calculate relative distances
        x0 = pipe0["x"] - self.playerx  # Horizontal distance to the next pipe
        y0 = pipe0["y"] - self.playery  # Vertical distance to the next lower pipe

        if -50 < x0 <= 0:
            y1 = pipe1["y"] - self.playery  # Vertical distance to the subsequent pipe
        else:
            y1 = 0

        # Discretization Grids (Bucketing)
        # Using modulo arithmetic to group close coordinates into the same state bin.

        # Discretize horizontal distance (x0)
        if x0 < -40:
            x0 = int(x0)
        elif x0 < 140:
            x0 = int(x0) - (int(x0) % 10)  # 10-pixel resolution for close pipes
        else:
            x0 = int(x0) - (int(x0) % 70)  # 70-pixel resolution for distant pipes

        # Discretize vertical distances (y0 and y1)
        if -180 < y0 < 180:
            y0 = int(y0) - (int(y0) % 10)
        else:
            y0 = int(y0) - (int(y0) % 60)

        if -180 < y1 < 180:
            y1 = int(y1) - (int(y1) % 10)
        else:
            y1 = int(y1) - (int(y1) % 60)

        vely = int(self.playerVelY)  # Bird's vertical velocity

        # Combine variables into a unique string representing the discrete Markov state
        return f"{x0}_{y0}_{vely}_{y1}"

    def _get_random_pipe(self) -> List[Dict[str, float]]:
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.cfg.pipe_gap_size))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.cfg.pipe_h
        pipeX = self.cfg.screen_width + 10
        return [
            {"x": pipeX, "y": gapY - pipeHeight},
            {"x": pipeX, "y": gapY + self.cfg.pipe_gap_size},
        ]

    def _check_crash(self, player: Dict[str, Any], upperPipes: List[Dict[str, float]],
                     lowerPipes: List[Dict[str, float]]) -> Tuple[bool, bool]:
        pi = player["index"]
        player["w"] = self.cfg.player_w
        player["h"] = self.cfg.player_h

        if (player["y"] + player["h"] >= self.BASEY - 1) or (player["y"] + player["h"] <= 0):
            return True, True

        use_pixel = self.HITMASKS is not None
        playerRect = pygame.Rect(player["x"], player["y"], player["w"], player["h"])
        pipeW = self.cfg.pipe_w
        pipeH = self.cfg.pipe_h

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(uPipe["x"], uPipe["y"], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe["x"], lPipe["y"], pipeW, pipeH)

            if not use_pixel:
                if playerRect.colliderect(uPipeRect) or playerRect.colliderect(lPipeRect):
                    return True, False
                continue

            pHitMask = self.HITMASKS["player"][pi]
            uHitmask = self.HITMASKS["pipe"][0]
            lHitmask = self.HITMASKS["pipe"][1]

            uCollide = self._pixel_collision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = self._pixel_collision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True, False

        return False, False

    @staticmethod
    def _pixel_collision(rect1: pygame.Rect, rect2: pygame.Rect, hitmask1, hitmask2) -> bool:
        rect = rect1.clip(rect2)
        if rect.width == 0 or rect.height == 0:
            return False
        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y
        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def _snapshot_info(self, groundCrash: bool) -> Dict[str, Any]:
        return {
            "groundCrash": groundCrash,
            "basex": self.basex,
            "upperPipes": self.upperPipes,
            "lowerPipes": self.lowerPipes,
            "score": self.score,
            "playerVelY": self.playerVelY,
            "y": self.playery,
        }