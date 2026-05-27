# /// script
# dependencies = [
#   "numpy>=1.26",
#   "playwright>=1.40",
#   "rich>=13.0",
#   "torch>=2.2",
# ]
# ///
"""Train a tiny DQN policy for Turbo Kart Dash.

Run with uv:
  uv run train_dqn.py --steps 50000 --episodes-eval 20 --out dqn_model.json

The exported JSON can be injected into the game as window.HEADLESS_DQN_WEIGHTS
and used by the headless/browser policy hook with agent=dqn.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.data, batch_size)
        obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones


class DQN(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TurboKartEnv:
    def __init__(
        self,
        page: Page,
        index_path: Path,
        map_id: str,
        character: str,
        frames: int,
        solo: bool,
        no_items: bool,
        no_hazards: bool,
        frame_stack: int = 1,
        frame_skip: int = 4,
        opponent_models: list[dict[str, Any]] | None = None,
    ):
        self.page = page
        flags = [
            "headless=1",
            "external=1",
            f"map={map_id}",
            f"char={character}",
            f"frames={frames}",
            f"solo={1 if solo else 0}",
            f"noItems={1 if no_items else 0}",
            f"noHazards={1 if no_hazards else 0}",
        ]
        self.url = index_path.resolve().as_uri() + "?" + "&".join(flags)
        self.map_id = map_id
        self.character = character
        self.frames = frames
        self.solo = solo
        self.no_items = no_items
        self.no_hazards = no_hazards
        self.frame_stack = max(1, int(frame_stack))
        self.frame_skip = max(1, int(frame_skip))
        self.opponent_models = opponent_models or []
        self.obs_keys: list[str] = []
        self.actions: list[dict[str, Any]] = []
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def _stack_keys(self, keys: list[str]) -> list[str]:
        if self.frame_stack <= 1:
            return keys
        stacked = []
        for lag in range(self.frame_stack):
            suffix = "" if lag == 0 else f"@-{lag}"
            stacked.extend(f"{key}{suffix}" for key in keys)
        return stacked

    def _stack_obs(self, obs: np.ndarray, reset: bool = False) -> np.ndarray:
        if reset or not self._frames:
            self._frames.clear()
            for _ in range(self.frame_stack):
                self._frames.append(obs.copy())
        else:
            self._frames.appendleft(obs.copy())
            while len(self._frames) < self.frame_stack:
                self._frames.append(obs.copy())
        if self.frame_stack <= 1:
            return obs
        return np.concatenate(list(self._frames)).astype(np.float32)

    def load(self) -> None:
        self.page.goto(self.url, wait_until="load")
        ready = self.page.evaluate("window.__HEADLESS_READY__")
        if not ready:
            raise RuntimeError("Headless RL API did not initialize")

    def reset(self) -> np.ndarray:
        return self.reset_with()

    def reset_with(
        self,
        *,
        map_id: str | None = None,
        character: str | None = None,
        opponent_models: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        if map_id is not None:
            self.map_id = map_id
        if character is not None:
            self.character = character
        if opponent_models is not None:
            self.opponent_models = opponent_models
        result = self.page.evaluate(
            """(cfg) => window.rlReset(cfg)""",
            {
                "map": self.map_id,
                "character": self.character,
                "frames": self.frames,
                "solo": self.solo,
                "noItems": self.no_items,
                "noHazards": self.no_hazards,
                "frameSkip": self.frame_skip,
                "opponentModels": self.opponent_models,
            },
        )
        self.obs_keys = self._stack_keys(result["obsKeys"])
        self.actions = result["actions"]
        return self._stack_obs(np.asarray(result["obs"], dtype=np.float32), reset=True)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        result = self.page.evaluate("(a) => window.rlStep(a)", int(action))
        obs = self._stack_obs(np.asarray(result["obs"], dtype=np.float32))
        reward = float(result["reward"])
        done = bool(result["done"])
        info = dict(result["info"])
        return obs, reward, done, info


def smoothgrad_attribution(
    model: DQN,
    buffer: ReplayBuffer,
    obs_keys: list[str],
    *,
    n_samples: int = 200,
    n_smooth: int = 30,
    noise_std: float = 0.1,
) -> dict[str, float]:
    if len(buffer) < n_samples:
        return {}
    batch = random.sample(buffer.data, n_samples)
    obs_batch = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32)
    obs_batch.requires_grad_(True)

    attributions = torch.zeros(obs_batch.shape[1])
    for _ in range(n_smooth):
        noisy = (obs_batch + torch.randn_like(obs_batch) * noise_std).detach()
        noisy.requires_grad_(True)
        q_values = model(noisy)
        best_q = q_values.max(dim=1).values.sum()
        best_q.backward()
        if noisy.grad is not None:
            attributions += noisy.grad.abs().mean(dim=0).detach()
        model.zero_grad()

    attributions /= max(1, n_smooth)
    result: dict[str, float] = {}
    for i, score in enumerate(attributions.tolist()):
        key = obs_keys[i] if i < len(obs_keys) else f"obs_{i}"
        result[key] = round(score, 6)
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def print_attribution_table(
    console: Console,
    title: str,
    attribution: dict[str, float],
    top_n: int = 10,
    bottom_n: int = 10,
) -> None:
    if not attribution:
        return
    sorted_items = list(attribution.items())
    max_val = max(v for _, v in sorted_items) if sorted_items else 1
    table = Table(title=title)
    table.add_column("Feature", style="cyan")
    table.add_column("Attention", justify="right", style="yellow")
    table.add_column("Bar", style="green")
    top_items = sorted_items[:top_n]
    bottom_items = sorted_items[-bottom_n:] if len(sorted_items) > top_n + bottom_n else []
    for key, score in top_items:
        bar_len = int(24 * score / max(max_val, 1e-9))
        table.add_row(key, f"{score:.4f}", "█" * bar_len)
    if bottom_items:
        table.add_row("···", "", "", style="dim")
        for key, score in bottom_items:
            bar_len = int(24 * score / max(max_val, 1e-9))
            table.add_row(key, f"{score:.4f}", "█" * bar_len, style="dim")
    total = len(sorted_items)
    console.print(table)
    console.print(f"  [dim]{total} features total · showing top {min(top_n, total)} + bottom {min(bottom_n, len(bottom_items))}[/dim]")


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    t = min(1.0, step / max(1, decay_steps))
    return start + (end - start) * t


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_metrics_table(console: Console, title: str, metrics: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, format_float(value, 4))
        else:
            table.add_row(key, str(value))
    console.print(table)


def print_eval_report(
    console: Console,
    title: str,
    report: dict[str, Any],
    reference: dict[str, dict[str, float]],
) -> None:
    table = Table(title=title)
    table.add_column("Track", style="cyan")
    table.add_column("Solo F", justify="right", style="green")
    table.add_column("Solo R", justify="right")
    table.add_column("Classic F", justify="right", style="green")
    table.add_column("Classic R", justify="right")
    table.add_column("Classic Laps", justify="right")
    table.add_column("Classic Win", justify="right", style="bold green")
    table.add_column("Classic Winner", justify="right")
    table.add_column("Ref R", justify="right", style="magenta")
    table.add_column("Ref F", justify="right", style="magenta")
    for track, metrics in report.get("tracks", {}).items():
        solo = metrics.get("solo", {})
        classic = metrics.get("classic", {})
        ref = reference.get(track, {})
        table.add_row(
            track,
            format_float(solo.get("finish_rate"), 2),
            format_float(solo.get("avg_reward"), 1),
            format_float(classic.get("finish_rate"), 2),
            format_float(classic.get("avg_reward"), 1),
            format_float(classic.get("avg_laps"), 2),
            format_float(classic.get("player_win_rate"), 2),
            ", ".join(f"{k}:{v}" for k, v in (classic.get("winner_chars") or {}).items()) or "-",
            format_float(ref.get("avg_reward"), 1),
            format_float(ref.get("finish_rate"), 2),
        )
    console.print(table)


def launch_chromium(playwright: Any, auto_install: bool) -> Any:
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError as exc:
        message = str(exc)
        missing_browser = "Executable doesn't exist" in message or "playwright install" in message
        if not missing_browser or not auto_install:
            if missing_browser:
                raise RuntimeError(
                    "Playwright is installed, but its Chromium browser is missing. "
                    "Run this from turbokart:\n\n"
                    "  uv run train_dqn.py --install-browser-only\n\n"
                    "Then rerun the trainer."
                ) from None
            raise

        print(json.dumps({"event": "browser_install", "command": "python -m playwright install chromium"}), flush=True)
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        return playwright.chromium.launch(headless=True)


def export_dqn_json(
    model: DQN,
    obs_keys: list[str],
    actions: list[dict[str, Any]],
    out_path: Path,
    meta: dict[str, Any],
    manifest_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    layers: list[dict[str, Any]] = []
    for i, layer in enumerate(linear_layers):
        layers.append(
            {
                "weights": layer.weight.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
                "biases": layer.bias.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
                "activation": "linear" if i == len(linear_layers) - 1 else "tanh",
            }
        )

    payload = {
        "type": "dqn",
        "format": "turbo-kart-headless-dqn-v1",
        "observationKeys": obs_keys,
        "actions": actions,
        "layers": layers,
        "meta": meta,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if manifest_path is not None:
        update_model_manifest(manifest_path, out_path, payload)


def update_model_manifest(manifest_path: Path, model_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    models = manifest.get("models")
    if not isinstance(models, list):
        models = []
    model_id = payload["meta"].get("id") or model_path.stem
    try:
        rel_path = model_path.relative_to(manifest_path.parent.parent).as_posix()
    except ValueError:
        rel_path = model_path.as_posix()
    entry = {
        "id": model_id,
        "name": payload["meta"].get("name") or model_id,
        "path": rel_path,
        "map": payload["meta"].get("map"),
        "character": payload["meta"].get("character"),
        "format": payload.get("format"),
        "observationKeyCount": len(payload.get("observationKeys", [])),
        "actionCount": len(payload.get("actions", [])),
        "metrics": payload["meta"].get("metrics", {}),
        "eval": payload["meta"].get("eval", {}),
        "frameStack": payload["meta"].get("frameStack", 1),
        "frameSkip": payload["meta"].get("frameSkip", 1),
        "updatedAt": int(time.time()),
    }
    models = [m for m in models if m.get("id") != model_id]
    models.append(entry)
    manifest["models"] = sorted(models, key=lambda m: m.get("updatedAt", 0), reverse=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_league_models(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    root = manifest_path.parent.parent
    models = manifest.get("models", [])
    loaded = []
    for idx, entry in enumerate(models):
        path = entry.get("path")
        if not path:
            continue
        model_path = root / path
        if not model_path.exists():
            continue
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "dqn" or not payload.get("layers"):
            continue
        loaded.append({"entry": entry, "payload": payload, "rank": idx})
        if limit is not None and len(loaded) >= limit:
            break
    return loaded


def sample_league_opponents(args: argparse.Namespace, league: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not args.self_play or not league:
        return []
    opponents = []
    for _ in range(args.league_opponents):
        if random.random() < args.classic_opponent_prob:
            continue
        weights = []
        for item in league:
            updated = float(item["entry"].get("updatedAt", 0) or 0)
            rank_weight = math.exp(-item["rank"] / max(0.001, args.league_recency_tau))
            weights.append(max(0.0001, rank_weight * (1.0 + updated * 0.0)))
        chosen = random.choices(league, weights=weights, k=1)[0]
        opponents.append(chosen["payload"])
    return opponents


def sample_character(args: argparse.Namespace) -> str:
    if not args.random_character:
        return args.character
    return random.choice(parse_csv(args.characters))


def sample_map(args: argparse.Namespace) -> str:
    if not args.random_map:
        return args.map
    return random.choice(parse_csv(args.maps))


@torch.no_grad()
def evaluate(
    env: TurboKartEnv,
    model: DQN,
    episodes: int,
    *,
    map_id: str,
    character: str,
    characters: list[str] | None = None,
    solo: bool,
    opponent_models: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    old_solo = env.solo
    old_opponents = env.opponent_models
    env.solo = solo
    env.opponent_models = opponent_models or []
    finishes = 0
    rewards = []
    laps = []
    race_times = []
    progresses = []
    winner_chars: dict[str, int] = {}
    player_wins = 0
    try:
        for _ in range(episodes):
            eval_char = random.choice(characters) if characters else character
            obs = env.reset_with(map_id=map_id, character=eval_char, opponent_models=env.opponent_models)
            done = False
            total_reward = 0.0
            last_info: dict[str, Any] = {}
            while not done:
                q = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())
                obs, reward, done, last_info = env.step(action)
                total_reward += reward
            finishes += int(bool(last_info.get("finished")))
            ranking = env.page.evaluate(
                """() => (
                    window.__lastRlRanking ||
                    (typeof rankAll === 'function'
                      ? rankAll().map(k => ({ name: k.name, charId: k.charId, finished: !!k.finished }))
                      : [])
                )"""
            )
            if ranking:
                winner = ranking[0].get("charId") or ranking[0].get("name") or "unknown"
                winner_chars[winner] = winner_chars.get(winner, 0) + 1
                if winner == eval_char:
                    player_wins += 1
            rewards.append(total_reward)
            laps.append(float(last_info.get("lap", 0)))
            race_times.append(float(last_info.get("raceTime", 0)))
            progresses.append(float(last_info.get("progress", 0)))
    finally:
        env.solo = old_solo
        env.opponent_models = old_opponents

    return {
        "episodes": episodes,
        "finish_rate": finishes / max(1, episodes),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_laps": float(np.mean(laps)) if laps else 0.0,
        "avg_race_time": float(np.mean(race_times)) if race_times else 0.0,
        "avg_progress": float(np.mean(progresses)) if progresses else 0.0,
        "winner_chars": winner_chars,
        "player_win_rate": player_wins / max(1, episodes),
    }


def evaluate_tracks(env: TurboKartEnv, model: DQN, args: argparse.Namespace) -> dict[str, Any]:
    per_track: dict[str, Any] = {}
    eval_maps = parse_csv(args.eval_maps)
    for map_id in eval_maps:
        char = args.character
        eval_chars = parse_csv(args.characters) if args.random_character else None
        per_track[map_id] = {
            "solo": evaluate(
                env,
                model,
                args.episodes_eval,
                map_id=map_id,
                character=char,
                characters=eval_chars,
                solo=True,
                opponent_models=[],
            ),
            "classic": evaluate(
                env,
                model,
                args.episodes_eval,
                map_id=map_id,
                character=char,
                characters=eval_chars,
                solo=False,
                opponent_models=[],
            ),
        }
    avg_reward = float(np.mean([m["classic"]["avg_reward"] for m in per_track.values()])) if per_track else 0.0
    avg_finish = float(np.mean([m["classic"]["finish_rate"] for m in per_track.values()])) if per_track else 0.0
    return {"avg_reward": avg_reward, "avg_finish_rate": avg_finish, "tracks": per_track}


def run_headless_waypoint_reference(
    page: Page,
    index_path: Path,
    *,
    map_id: str,
    character: str,
    frames: int,
    episodes: int,
    solo: bool,
    no_items: bool,
    no_hazards: bool,
) -> dict[str, float]:
    flags = [
        "headless=1",
        "agent=waypoint",
        f"map={map_id}",
        f"char={character}",
        f"frames={frames}",
        f"episodes={episodes}",
        f"solo={1 if solo else 0}",
        f"noItems={1 if no_items else 0}",
        f"noHazards={1 if no_hazards else 0}",
    ]
    page.goto(index_path.resolve().as_uri() + "?" + "&".join(flags), wait_until="load")
    result = page.evaluate("window.__HEADLESS_RESULT__")
    aggregate = result.get("aggregate", {})
    return {
        "episodes": episodes,
        "finish_rate": float(aggregate.get("finishCount", 0)) / max(1, episodes),
        "avg_reward": float(aggregate.get("avgReward", 0)),
        "avg_laps": float(aggregate.get("totalPlayerLaps", 0)) / max(1, episodes),
        "avg_progress": float(aggregate.get("avgPlayerProgress", 0)),
    }


def waypoint_references(browser: Any, index_path: Path, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    page = browser.new_page()
    refs = {}
    for map_id in parse_csv(args.eval_maps):
        refs[map_id] = run_headless_waypoint_reference(
            page,
            index_path,
            map_id=map_id,
            character=args.character,
            frames=args.frames,
            episodes=args.reference_episodes,
            solo=args.solo,
            no_items=args.no_items,
            no_hazards=args.no_hazards,
        )
    page.close()
    return refs


def train(args: argparse.Namespace) -> None:
    console = Console()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    index_path = Path(args.index).resolve()
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    with sync_playwright() as p:
        browser = launch_chromium(p, args.auto_install_browser)
        page = browser.new_page()
        env = TurboKartEnv(
            page=page,
            index_path=index_path,
            map_id=args.map,
            character=args.character,
            frames=args.frames,
            solo=args.solo,
            no_items=args.no_items,
            no_hazards=args.no_hazards,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
        )
        env.load()
        league = load_league_models(Path(args.league_manifest), args.league_limit) if args.self_play else []
        obs = env.reset_with(
            map_id=sample_map(args),
            character=sample_character(args),
            opponent_models=sample_league_opponents(args, league),
        )
        eval_page = browser.new_page()
        eval_env = TurboKartEnv(
            page=eval_page,
            index_path=index_path,
            map_id=args.map,
            character=args.character,
            frames=args.frames,
            solo=args.solo,
            no_items=args.no_items,
            no_hazards=args.no_hazards,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
        )
        eval_env.load()
        obs_dim = int(obs.shape[0])
        action_dim = len(env.actions)

        q = DQN(obs_dim, action_dim, args.hidden)
        target_q = DQN(obs_dim, action_dim, args.hidden)
        target_q.load_state_dict(q.state_dict())
        optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)
        buffer = ReplayBuffer(args.buffer_size)

        episode_reward = 0.0
        episode_count = 0
        best_eval_reward = -math.inf
        last_loss = None
        started_at = time.perf_counter()
        recent_rewards: deque[float] = deque(maxlen=20)
        recent_laps: deque[float] = deque(maxlen=20)
        recent_finishes: deque[float] = deque(maxlen=20)
        recent_maps: deque[str] = deque(maxlen=20)
        recent_chars: deque[str] = deque(maxlen=20)
        action_counts = np.zeros(action_dim, dtype=np.int64)
        recent_action_counts = np.zeros(action_dim, dtype=np.int64)
        recent_q_max: deque[float] = deque(maxlen=1000)
        recent_q_mean: deque[float] = deque(maxlen=1000)
        reference_metrics = waypoint_references(browser, index_path, args) if args.reference_episodes > 0 else {}

        start_payload = {
            "event": "start",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "steps": args.steps,
            "map": args.map,
            "character": args.character,
            "solo": args.solo,
            "no_items": args.no_items,
            "no_hazards": args.no_hazards,
            "frame_stack": args.frame_stack,
            "frame_skip": args.frame_skip,
        }
        if args.json_logs:
            emit_json(start_payload)
            progress = None
            task_id = None
        else:
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"[bold]Map[/bold]: {args.map}",
                            f"[bold]Base character[/bold]: {args.character}",
                            "[bold]Training characters[/bold]: "
                            f"{args.characters if args.random_character else args.character}",
                            f"[bold]Observations[/bold]: {obs_dim}",
                            f"[bold]Actions[/bold]: {action_dim}",
                            f"[bold]Steps[/bold]: {args.steps}",
                            f"[bold]Eval maps[/bold]: {args.eval_maps}",
                            f"[bold]Training maps[/bold]: {args.maps if args.random_map else args.map}",
                            f"[bold]Random map[/bold]: {args.random_map}",
                            f"[bold]Random character[/bold]: {args.random_character}",
                            f"[bold]Frame stack[/bold]: {args.frame_stack}",
                            f"[bold]Frame skip[/bold]: {args.frame_skip}",
                            f"[bold]Self-play[/bold]: {args.self_play} ({len(league)} league models)",
                        ]
                    ),
                    title="TurboKart DQN Training",
                    border_style="cyan",
                )
            )
            progress = Progress(
                TextColumn("[bold cyan]training"),
                BarColumn(),
                TextColumn("{task.percentage:>5.1f}%"),
                TextColumn("step {task.completed:.0f}/{task.total:.0f}"),
                TextColumn("eps {task.fields[eps]}"),
                TextColumn("loss {task.fields[loss]}"),
                TextColumn("ep {task.fields[episodes]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            progress.start()
            task_id = progress.add_task(
                "train",
                total=args.steps,
                eps=f"{args.eps_start:.3f}",
                loss="-",
                episodes="0",
            )

        for step in range(1, args.steps + 1):
            eps = epsilon_by_step(step, args.eps_start, args.eps_end, args.eps_decay)
            if random.random() < eps:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    recent_q_max.append(float(torch.max(q_values).detach().cpu().item()))
                    recent_q_mean.append(float(torch.mean(q_values).detach().cpu().item()))
                    action = int(torch.argmax(q_values, dim=1).item())
            action_counts[action] += 1
            recent_action_counts[action] += 1

            next_obs, reward, done, info = env.step(action)
            buffer.add(Transition(obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward

            if done:
                episode_count += 1
                if episode_count % args.log_every_episodes == 0:
                    episode_payload = {
                        "event": "episode",
                        "episode": episode_count,
                        "step": step,
                        "epsilon": round(eps, 4),
                        "episode_reward": round(episode_reward, 3),
                        "lap": info.get("lap"),
                        "finished": info.get("finished"),
                        "progress": round(float(info.get("progress", 0)), 3),
                    }
                    if args.json_logs:
                        emit_json(episode_payload)
                    else:
                        console.print(
                            "[magenta]episode[/magenta] "
                            f"#{episode_count} step={step} reward={episode_payload['episode_reward']} "
                            f"lap={info.get('lap')} finished={info.get('finished')} "
                            f"progress={episode_payload['progress']}"
                        )
                recent_rewards.append(episode_reward)
                recent_laps.append(float(info.get("lap", 0)))
                recent_finishes.append(1.0 if info.get("finished") else 0.0)
                recent_maps.append(env.map_id)
                recent_chars.append(env.character)
                if args.self_play:
                    league = load_league_models(Path(args.league_manifest), args.league_limit)
                obs = env.reset_with(
                    map_id=sample_map(args),
                    character=sample_character(args),
                    opponent_models=sample_league_opponents(args, league),
                )
                episode_reward = 0.0

            if len(buffer) >= args.batch_size and step >= args.learning_starts:
                batch = buffer.sample(args.batch_size)
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch
                with torch.no_grad():
                    next_q = target_q(b_next_obs).max(dim=1).values
                    target = b_rewards + args.gamma * (1.0 - b_dones) * next_q
                pred = q(b_obs).gather(1, b_actions).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(pred, target)
                last_loss = float(loss.detach().cpu().item())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), args.grad_clip)
                optimizer.step()

            if step % args.target_update == 0:
                target_q.load_state_dict(q.state_dict())

            if step % args.log_every_steps == 0:
                elapsed = max(1e-6, time.perf_counter() - started_at)
                progress_payload = {
                    "event": "progress",
                    "step": step,
                    "steps": args.steps,
                    "pct": round(step / args.steps * 100, 2),
                    "steps_per_sec": round(step / elapsed, 2),
                    "episodes": episode_count,
                    "epsilon": round(eps, 4),
                    "buffer": len(buffer),
                    "loss": None if last_loss is None else round(last_loss, 6),
                    "recent_avg_reward": round(float(np.mean(recent_rewards)), 3) if recent_rewards else None,
                    "recent_avg_laps": round(float(np.mean(recent_laps)), 3) if recent_laps else None,
                    "recent_finish_rate": round(float(np.mean(recent_finishes)), 3) if recent_finishes else None,
                    "recent_q_max": round(float(np.mean(recent_q_max)), 3) if recent_q_max else None,
                    "recent_q_mean": round(float(np.mean(recent_q_mean)), 3) if recent_q_mean else None,
                    "top_action": env.actions[int(np.argmax(recent_action_counts))]["name"]
                    if recent_action_counts.sum() > 0
                    else None,
                    "recent_maps": dict(sorted({m: recent_maps.count(m) for m in set(recent_maps)}.items())),
                    "recent_chars": dict(sorted({c: recent_chars.count(c) for c in set(recent_chars)}.items())),
                }
                recent_action_counts[:] = 0
                if args.json_logs:
                    emit_json(progress_payload)
                elif progress is not None and task_id is not None:
                    top_action = progress_payload["top_action"] or "-"
                    progress.update(
                        task_id,
                        completed=step,
                        eps=f"{eps:.3f}",
                        loss=format_float(last_loss, 4),
                        episodes=f"{episode_count} a:{top_action}",
                    )

            if step % args.eval_every == 0:
                eval_report = evaluate_tracks(eval_env, q, args)
                track_report = eval_report["tracks"].get(args.map) or next(iter(eval_report["tracks"].values()))
                metrics = track_report["classic"]
                attribution = smoothgrad_attribution(q, buffer, env.obs_keys) if len(buffer) >= 200 else {}
                if attribution:
                    eval_report["attribution"] = attribution
                if args.json_logs:
                    emit_json({"event": "eval", "eval_step": step, **eval_report, "reference": reference_metrics})
                else:
                    print_eval_report(console, f"Evaluation @ step {step}", eval_report, reference_metrics)
                    if attribution:
                        print_attribution_table(console, f"SmoothGrad Attribution @ step {step}", attribution)
                checkpoint_path = Path(args.checkpoint_dir) / f"{args.model_id}-step-{step}.json"
                export_dqn_json(
                    q,
                    env.obs_keys,
                    env.actions,
                    checkpoint_path,
                    {
                        "id": f"{args.model_id}-step-{step}",
                        "name": f"{args.model_name} step {step}",
                        "step": step,
                        "map": args.map,
                        "character": args.character,
                            "frameStack": args.frame_stack,
                            "frameSkip": args.frame_skip,
                        "metrics": metrics,
                        "eval": eval_report,
                        "reference": reference_metrics,
                        "attribution": attribution,
                    },
                    # Checkpoints stay local and ignored; only final models update the public manifest.
                    None,
                )
                if eval_report["avg_reward"] > best_eval_reward:
                    best_eval_reward = eval_report["avg_reward"]

        final_eval_report = evaluate_tracks(eval_env, q, args)
        final_track_report = final_eval_report["tracks"].get(args.map) or next(
            iter(final_eval_report["tracks"].values())
        )
        final_metrics = final_track_report["classic"]
        final_attribution = smoothgrad_attribution(q, buffer, env.obs_keys) if len(buffer) >= 200 else {}
        if final_attribution:
            final_eval_report["attribution"] = final_attribution
        export_dqn_json(
            q,
            env.obs_keys,
            env.actions,
            Path(args.out),
            {
                "id": args.model_id,
                "name": args.model_name,
                "step": args.steps,
                "map": args.map,
                "character": args.character,
                "frameStack": args.frame_stack,
                "frameSkip": args.frame_skip,
                "metrics": final_metrics,
                "eval": final_eval_report,
                "reference": reference_metrics,
                "attribution": final_attribution,
            },
            Path(args.manifest),
        )
        if progress is not None:
            progress.update(task_id, completed=args.steps)
            progress.stop()
        if args.json_logs:
            print(
                json.dumps({"event": "final_eval", "final_eval": final_metrics, "out": args.out}, indent=2),
                flush=True,
            )
        else:
            if episode_count == 0:
                console.print(
                    "[yellow]No training episode completed before the step budget ended. "
                    "Increase --steps or reduce --frames for more episode-level feedback.[/yellow]"
                )
            print_eval_report(console, "Final Evaluation", final_eval_report, reference_metrics)
            if final_attribution:
                print_attribution_table(console, "Final SmoothGrad Attribution", final_attribution)
            console.print(f"[green]Exported model:[/green] {args.out}")
        eval_page.close()
        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="index.html")
    parser.add_argument("--out", default="models/dqn_model.json")
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--model-id", default="dqn-core-mainframe")
    parser.add_argument("--model-name", default="DQN Core Mainframe")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--map", default="core_mainframe")
    parser.add_argument(
        "--maps",
        default="core_mainframe,audit_super_ring,compliance_chicane,black_ice_data_vault,protocol_amendment_labyrinth",
    )
    parser.add_argument("--random-map", action="store_true")
    parser.add_argument("--eval-maps", default="core_mainframe")
    parser.add_argument("--character", default="florian")
    parser.add_argument("--characters", default="anton,artur,rissal,pia,florian")
    parser.add_argument("--random-character", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frames", type=int, default=7200)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--episodes-eval", type=int, default=10)
    parser.add_argument("--reference-episodes", type=int, default=2)
    parser.add_argument("--log-every-episodes", type=int, default=10)
    parser.add_argument("--log-every-steps", type=int, default=1_000)
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON lines instead of Rich progress output")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--solo", action="store_true", default=True)
    parser.add_argument("--with-opponents", dest="solo", action="store_false")
    parser.add_argument("--no-items", action="store_true", default=True)
    parser.add_argument("--with-items", dest="no_items", action="store_false")
    parser.add_argument("--no-hazards", action="store_true", default=True)
    parser.add_argument("--with-hazards", dest="no_hazards", action="store_false")
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--league-manifest", default="models/manifest.json")
    parser.add_argument("--league-limit", type=int, default=16)
    parser.add_argument("--league-opponents", type=int, default=3)
    parser.add_argument("--league-recency-tau", type=float, default=4.0)
    parser.add_argument("--classic-opponent-prob", type=float, default=0.25)
    parser.add_argument("--no-auto-install-browser", dest="auto_install_browser", action="store_false")
    parser.add_argument("--install-browser-only", action="store_true")
    parser.set_defaults(auto_install_browser=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.install_browser_only:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    else:
        train(parsed_args)
