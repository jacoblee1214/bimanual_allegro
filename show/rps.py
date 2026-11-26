"""
rps.py — rock-paper-scissors against the robot: the logic, without ROS.

The visitor's hand comes from the tracker's 21 world landmarks (MediaPipe order, metres).
A finger counts as extended by how far its tip is from the wrist, relative to its knuckle:
1.5 to 2.0 when straight and 0.8 to 1.1 when curled in the recorded clips (data/results/
m5_detections_*.npz), whatever the hand's size or direction.

Two ways for the robot to pick its throw (Game.mode):
  fair    drawn at random when the round starts, shown on "SHOOT"
  reader  the robot watches the visitor's fingers start to open and throws what beats it;
          the time from the first finger motion to the decision is reported (react_ms)
The round is judged the same way in both: the visitor's throw is the first gesture that stays
the same for `stable` frames after "SHOOT".

Robot hand poses are in the hand controller's joint order (thumb, index, middle, ring, pinky;
each abduction, then three flexions). ROCK is what the real left hand reached as a fist in
data/runs/t5 (2026-11-26), with the abductions set to 0; PAPER is the open hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROCK, PAPER, SCISSORS = "rock", "paper", "scissors"
BEATS = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}      # BEATS[x] wins against x
TIPS, KNUCKLES = (8, 12, 16, 20), (5, 9, 13, 17)            # index, middle, ring, pinky

_THUMB_FIST = [0.63, -1.38, 0.74, 0.96]
_CURLED, _STRAIGHT = [0.0, 1.3, 1.3, 1.3], [0.0, 0.0, 0.0, 0.0]
POSES = {
    ROCK: np.array(_THUMB_FIST + _CURLED * 4),
    PAPER: np.zeros(20),
    SCISSORS: np.array(_THUMB_FIST + _STRAIGHT * 2 + _CURLED * 2),
}
POSES["relaxed"] = 0.35 * POSES[ROCK]


def extension(landmarks) -> np.ndarray:
    """Tip-to-wrist over knuckle-to-wrist distance for index, middle, ring, pinky."""
    p = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
    wrist = p[0]
    return (np.linalg.norm(p[list(TIPS)] - wrist, axis=1)
            / np.linalg.norm(p[list(KNUCKLES)] - wrist, axis=1))


def classify(ext: np.ndarray, threshold: float = 1.35) -> str | None:
    """The gesture of a settled hand, or None if it is none of the three."""
    index, middle, ring, pinky = (bool(v) for v in ext > threshold)
    count = index + middle + ring + pinky
    if count == 0:
        return ROCK
    if index and middle and count >= 3:
        return PAPER
    if index and middle and count == 2:
        return SCISSORS
    return None


def read_early(ext: np.ndarray, moving: float = 1.3, still: float = 1.2) -> str | None:
    """A guess from a fist that has just started to open: which throw is it becoming?"""
    index, middle, ring, pinky = ext
    if index > moving and middle > moving:
        if ring > moving or pinky > moving:
            return PAPER
        if ring < still and pinky < still:
            return SCISSORS
    return None


def outcome(robot: str, human: str | None) -> str:
    if human is None:
        return "no throw"
    if robot == human:
        return "draw"
    return "robot" if BEATS[human] == robot else "you"


@dataclass
class Game:
    """One round at a time. Call start(), then step(t, ext) with every tracker frame."""
    rng: np.random.Generator
    mode: str = "fair"
    period: float = 1.0             # s per count
    ready: float = 1.5              # s before the first count
    lead: float = 0.25              # fair: the robot starts its throw this long before SHOOT
    judge_until: float = 1.5        # s after SHOOT to wait for the visitor's throw
    result: float = 3.0             # s the result stays up
    stable: int = 4                 # frames a gesture must hold to count
    phase: str = "mirror"           # mirror, ready, count, shoot, result
    label: str = ""
    robot: str | None = None        # what the robot's hand shows now
    human: str | None = None
    winner: str | None = None
    react_ms: float | None = None
    score: dict = field(default_factory=lambda: {"robot": 0, "you": 0, "draw": 0})
    _t0: float = 0.0
    _choice: str | None = None
    _run: tuple = (None, 0)
    _early: tuple = (None, 0)
    _moved_at: float | None = None
    _thrown: bool = False
    _result_at: float = 0.0

    @property
    def shoot_at(self) -> float:
        return self._t0 + self.ready + 3 * self.period

    def start(self, t: float) -> None:
        self.phase, self._t0, self.label = "ready", t, "GET READY"
        self.robot, self.human, self.winner, self.react_ms = ROCK, None, None, None
        self._choice = str(self.rng.choice([ROCK, PAPER, SCISSORS])) if self.mode == "fair" else None
        self._run, self._early, self._moved_at, self._thrown = (None, 0), (None, 0), None, False

    def pump(self, t: float) -> float:
        """0..1: how far the robot's fist is down, three pumps during the count."""
        s = (t - self._t0 - self.ready) / self.period
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * s)) if self.phase == "count" and 0.0 <= s <= 3.0 else 0.0

    def step(self, t: float, ext: np.ndarray | None) -> None:
        if self.phase == "mirror":
            return
        shoot = self.shoot_at
        if self.phase == "ready" and t >= self._t0 + self.ready:
            self.phase = "count"
        if self.phase == "count":
            beat = int((t - self._t0 - self.ready) / self.period)
            self.label = ("ROCK", "PAPER", "SCISSORS")[min(beat, 2)]
            if t >= shoot:
                self.phase, self.label = "shoot", "SHOOT!"
        if self.phase in ("count", "shoot"):
            self._pick(t, ext, shoot)
        if self.phase == "shoot":
            self._judge(t, ext, shoot)
        if self.phase == "result" and t >= self._result_at + self.result:
            self.phase, self.label, self.robot = "mirror", "", None

    def _pick(self, t: float, ext, shoot: float) -> None:
        if self._thrown:
            return
        if self.mode == "fair":
            if t >= shoot - self.lead:
                self.robot, self._thrown = self._choice, True
            return
        if t < shoot - 0.4:
            return
        if ext is not None:
            if self._moved_at is None and ext.max() > 1.2:
                self._moved_at = t
            guess = read_early(ext)
            self._early = (guess, self._early[1] + 1) if guess and guess == self._early[0] else (guess, 1)
            if guess and self._early[1] >= 2:
                self.robot, self._thrown = BEATS[guess], True
                self.react_ms = 1000.0 * (t - (self._moved_at if self._moved_at is not None else t))
                return
        if t >= shoot + 0.45:               # the fist stayed a fist
            self.robot, self._thrown = PAPER, True

    def _judge(self, t: float, ext, shoot: float) -> None:
        seen = classify(ext) if ext is not None and t >= shoot + 0.15 else None
        self._run = (seen, self._run[1] + 1) if seen and seen == self._run[0] else (seen, 1 if seen else 0)
        settled = seen is not None and self._run[1] >= self.stable
        if not self._thrown or (not settled and t < shoot + self.judge_until):
            return
        self.human = seen if settled else None
        self.winner = outcome(self.robot, self.human)
        if self.winner in self.score:
            self.score[self.winner] += 1
        self.phase, self._result_at = "result", t
        self.label = {"robot": "ROBOT WINS", "you": "YOU WIN!", "draw": "DRAW", "no throw": "NO THROW?"}[self.winner]
