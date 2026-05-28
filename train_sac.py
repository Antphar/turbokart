# /// script
# dependencies = [
#   "numpy>=1.26",
#   "playwright>=1.40",
#   "rich>=13.0",
#   "torch>=2.2",
# ]
# ///
"""Train a SAC (Soft Actor-Critic) policy for Turbo Kart Dash.

Run with uv:
  uv run train_sac.py --random-map --random-character --with-opponents --with-items --self-play
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from rl_common import (
    ReplayBuffer,
    Transition,
    TurboKartEnv,
    format_float,
    launch_chromium,
    load_league_models,
    parse_csv,
    print_attribution_table,
    print_eval_report,
    sample_character,
    sample_league_opponents,
    sample_map,
    smoothgrad_attribution,
    update_model_manifest,
    waypoint_references,
)


class GaussianActor(torch.nn.Module):
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, action_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(hidden, action_dim)
        self.log_std_head = torch.nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        action = torch.cat([
            torch.tanh(x_t[:, :1]),
            torch.sigmoid(x_t[:, 1:]),
        ], dim=1)

        log_prob = normal.log_prob(x_t)
        log_prob[:, 0] -= torch.log(1 - action[:, 0].pow(2) + 1e-6)
        log_prob[:, 1:] -= torch.log(action[:, 1:] * (1 - action[:, 1:]) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, mean

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.cat([
            torch.tanh(mean[:, :1]),
            torch.sigmoid(mean[:, 1:]),
        ], dim=1)


class TwinCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=1)
        return self.q1(x), self.q2(x)


@torch.no_grad()
def evaluate_sac(
    env: TurboKartEnv,
    actor: GaussianActor,
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
    rewards: list[float] = []
    laps: list[float] = []
    race_times: list[float] = []
    progresses: list[float] = []
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
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor.deterministic(obs_tensor).squeeze(0).numpy()
                obs, reward, done, last_info = env.step(action.tolist())
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


def evaluate_tracks_sac(env: TurboKartEnv, actor: GaussianActor, args: argparse.Namespace) -> dict[str, Any]:
    per_track: dict[str, Any] = {}
    eval_maps = parse_csv(args.eval_maps)
    for map_id in eval_maps:
        char = args.character
        eval_chars = parse_csv(args.characters) if args.random_character else None
        per_track[map_id] = {
            "solo": evaluate_sac(
                env, actor, args.episodes_eval,
                map_id=map_id, character=char, characters=eval_chars,
                solo=True, opponent_models=[],
            ),
            "classic": evaluate_sac(
                env, actor, args.episodes_eval,
                map_id=map_id, character=char, characters=eval_chars,
                solo=False, opponent_models=[],
            ),
        }
    avg_reward = float(np.mean([m["classic"]["avg_reward"] for m in per_track.values()])) if per_track else 0.0
    avg_finish = float(np.mean([m["classic"]["finish_rate"] for m in per_track.values()])) if per_track else 0.0
    return {"avg_reward": avg_reward, "avg_finish_rate": avg_finish, "tracks": per_track}


def export_sac_json(
    actor: GaussianActor,
    obs_keys: list[str],
    out_path: Path,
    meta: dict[str, Any],
    manifest_path: Path | None = None,
) -> None:
    def _serialize_linear(layer: torch.nn.Linear) -> dict[str, Any]:
        return {
            "weights": layer.weight.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
            "biases": layer.bias.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
        }

    trunk_linears = [m for m in actor.trunk if isinstance(m, torch.nn.Linear)]
    trunk_layers: list[dict[str, Any]] = []
    for layer in trunk_linears:
        d = _serialize_linear(layer)
        d["activation"] = "relu"
        trunk_layers.append(d)

    payload: dict[str, Any] = {
        "type": "sac",
        "format": "turbo-kart-headless-sac-v1",
        "architecture": "gaussian_actor",
        "observationKeys": obs_keys,
        "actionNames": ["steer", "throttle", "brake", "drift", "use_item", "use_ultimate"],
        "trunk": trunk_layers,
        "mean_head": {**_serialize_linear(actor.mean_head), "activation": "linear"},
        "continuous": True,
        "meta": meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if manifest_path is not None:
        update_model_manifest(manifest_path, out_path, payload)


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


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


def train(args: argparse.Namespace) -> None:
    console = Console()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    action_dim = 6
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

        actor = GaussianActor(obs_dim, action_dim, args.hidden)
        critic = TwinCritic(obs_dim, action_dim, args.hidden)
        target_critic = TwinCritic(obs_dim, action_dim, args.hidden)
        target_critic.load_state_dict(critic.state_dict())

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)

        target_entropy = args.target_entropy
        if target_entropy is None:
            target_entropy = -float(action_dim)
        log_alpha = torch.tensor([math.log(args.init_alpha)], dtype=torch.float32, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = log_alpha.exp().item()

        buffer = ReplayBuffer(args.buffer_size)

        episode_reward = 0.0
        episode_count = 0
        best_eval_reward = -math.inf
        last_critic_loss = None
        last_actor_loss = None
        started_at = time.perf_counter()
        recent_rewards: deque[float] = deque(maxlen=20)
        recent_laps: deque[float] = deque(maxlen=20)
        recent_finishes: deque[float] = deque(maxlen=20)
        recent_maps: deque[str] = deque(maxlen=20)
        recent_chars: deque[str] = deque(maxlen=20)
        recent_alpha: deque[float] = deque(maxlen=1000)
        reference_metrics = waypoint_references(browser, index_path, args) if args.reference_episodes > 0 else {}

        start_payload = {
            "event": "start",
            "algorithm": "sac",
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
                    "\n".join([
                        "[bold]Algorithm[/bold]: SAC (Soft Actor-Critic)",
                        f"[bold]Map[/bold]: {args.map}",
                        f"[bold]Base character[/bold]: {args.character}",
                        f"[bold]Training characters[/bold]: "
                        f"{args.characters if args.random_character else args.character}",
                        f"[bold]Observations[/bold]: {obs_dim}",
                        f"[bold]Actions[/bold]: {action_dim} continuous",
                        f"[bold]Steps[/bold]: {args.steps}",
                        f"[bold]Eval maps[/bold]: {args.eval_maps}",
                        f"[bold]Training maps[/bold]: {args.maps if args.random_map else args.map}",
                        f"[bold]Random map[/bold]: {args.random_map}",
                        f"[bold]Random character[/bold]: {args.random_character}",
                        f"[bold]Frame stack[/bold]: {args.frame_stack}",
                        f"[bold]Frame skip[/bold]: {args.frame_skip}",
                        f"[bold]Hidden[/bold]: {args.hidden}",
                        f"[bold]Tau[/bold]: {args.tau}",
                        f"[bold]LR[/bold]: {args.lr}",
                        f"[bold]Target entropy[/bold]: {target_entropy}",
                        f"[bold]Initial alpha[/bold]: {args.init_alpha}",
                        f"[bold]Self-play[/bold]: {args.self_play} ({len(league)} league models)",
                    ]),
                    title="TurboKart SAC Training",
                    border_style="cyan",
                )
            )
            progress = Progress(
                TextColumn("[bold cyan]training"),
                BarColumn(),
                TextColumn("{task.percentage:>5.1f}%"),
                TextColumn("step {task.completed:.0f}/{task.total:.0f}"),
                TextColumn("α {task.fields[alpha]}"),
                TextColumn("c_loss {task.fields[c_loss]}"),
                TextColumn("a_loss {task.fields[a_loss]}"),
                TextColumn("ep {task.fields[episodes]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            progress.start()
            task_id = progress.add_task(
                "train",
                total=args.steps,
                alpha=f"{alpha:.3f}",
                c_loss="-",
                a_loss="-",
                episodes="0",
            )

        for step in range(1, args.steps + 1):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = actor.sample(obs_tensor)
            action_np = action_tensor.squeeze(0).numpy()

            next_obs, reward, done, info = env.step(action_np.tolist())
            buffer.add(Transition(obs, action_np.copy(), reward, next_obs, done))
            obs = next_obs
            episode_reward += reward

            if done:
                episode_count += 1
                if episode_count % args.log_every_episodes == 0:
                    episode_payload = {
                        "event": "episode",
                        "episode": episode_count,
                        "step": step,
                        "alpha": round(alpha, 4),
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
                            f"progress={episode_payload['progress']} α={alpha:.4f}"
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
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(args.batch_size)

                with torch.no_grad():
                    next_action, next_log_prob, _ = actor.sample(b_next_obs)
                    q1_next, q2_next = target_critic(b_next_obs, next_action)
                    q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                    target_q = b_rewards.unsqueeze(1) + args.gamma * (1 - b_dones.unsqueeze(1)) * q_next

                q1, q2 = critic(b_obs, b_actions)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                last_critic_loss = float(critic_loss.detach().item())

                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                critic_optimizer.step()

                new_action, log_prob, _ = actor.sample(b_obs)
                q1_new, q2_new = critic(b_obs, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_prob - q_new).mean()
                last_actor_loss = float(actor_loss.detach().item())

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                actor_optimizer.step()

                alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
                alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_optimizer.step()
                if args.min_alpha > 0:
                    with torch.no_grad():
                        log_alpha.clamp_(min=math.log(args.min_alpha))
                alpha = log_alpha.exp().item()
                recent_alpha.append(alpha)

                for tp, cp in zip(target_critic.parameters(), critic.parameters()):
                    tp.data.copy_(args.tau * cp.data + (1 - args.tau) * tp.data)

            if step % args.log_every_steps == 0:
                elapsed = max(1e-6, time.perf_counter() - started_at)
                progress_payload = {
                    "event": "progress",
                    "step": step,
                    "steps": args.steps,
                    "pct": round(step / args.steps * 100, 2),
                    "steps_per_sec": round(step / elapsed, 2),
                    "episodes": episode_count,
                    "alpha": round(alpha, 4),
                    "buffer": len(buffer),
                    "critic_loss": None if last_critic_loss is None else round(last_critic_loss, 6),
                    "actor_loss": None if last_actor_loss is None else round(last_actor_loss, 6),
                    "recent_avg_reward": round(float(np.mean(recent_rewards)), 3) if recent_rewards else None,
                    "recent_avg_laps": round(float(np.mean(recent_laps)), 3) if recent_laps else None,
                    "recent_finish_rate": round(float(np.mean(recent_finishes)), 3) if recent_finishes else None,
                    "recent_avg_alpha": round(float(np.mean(recent_alpha)), 4) if recent_alpha else None,
                    "recent_maps": dict(sorted({m: recent_maps.count(m) for m in set(recent_maps)}.items())),
                    "recent_chars": dict(sorted({c: recent_chars.count(c) for c in set(recent_chars)}.items())),
                }
                if args.json_logs:
                    emit_json(progress_payload)
                elif progress is not None and task_id is not None:
                    progress.update(
                        task_id,
                        completed=step,
                        alpha=f"{alpha:.3f}",
                        c_loss=format_float(last_critic_loss, 4),
                        a_loss=format_float(last_actor_loss, 4),
                        episodes=str(episode_count),
                    )

            if step % args.eval_every == 0:
                eval_report = evaluate_tracks_sac(eval_env, actor, args)
                track_report = eval_report["tracks"].get(args.map) or next(iter(eval_report["tracks"].values()))
                metrics = track_report["classic"]
                attribution = smoothgrad_attribution(
                    actor, buffer, env.obs_keys,
                    output_fn=lambda out: out[0].sum(),
                ) if len(buffer) >= 200 else {}
                if attribution:
                    eval_report["attribution"] = attribution
                if args.json_logs:
                    emit_json({"event": "eval", "eval_step": step, **eval_report, "reference": reference_metrics})
                else:
                    print_eval_report(console, f"Evaluation @ step {step}", eval_report, reference_metrics)
                    if attribution:
                        print_attribution_table(console, f"SmoothGrad Attribution @ step {step}", attribution)
                checkpoint_path = Path(args.checkpoint_dir) / f"{args.model_id}-step-{step}.json"
                export_sac_json(
                    actor, env.obs_keys, checkpoint_path,
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
                    None,
                )
                if eval_report["avg_reward"] > best_eval_reward:
                    best_eval_reward = eval_report["avg_reward"]

        final_eval_report = evaluate_tracks_sac(eval_env, actor, args)
        final_track_report = final_eval_report["tracks"].get(args.map) or next(
            iter(final_eval_report["tracks"].values())
        )
        final_metrics = final_track_report["classic"]
        final_attribution = smoothgrad_attribution(
            actor, buffer, env.obs_keys,
            output_fn=lambda out: out[0].sum(),
        ) if len(buffer) >= 200 else {}
        if final_attribution:
            final_eval_report["attribution"] = final_attribution
        export_sac_json(
            actor, env.obs_keys, Path(args.out),
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
    parser.add_argument("--out", default="models/sac-latest.json")
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--model-id", default="sac-latest")
    parser.add_argument("--model-name", default="SAC Latest")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--map", default="core_mainframe")
    parser.add_argument(
        "--maps",
        default="core_mainframe,audit_super_ring,compliance_chicane,black_ice_data_vault,protocol_amendment_labyrinth",
    )
    parser.add_argument("--random-map", action="store_true")
    parser.add_argument(
        "--eval-maps",
        default="core_mainframe,audit_super_ring,compliance_chicane,black_ice_data_vault,protocol_amendment_labyrinth",
    )
    parser.add_argument("--character", default="florian")
    parser.add_argument("--characters", default="anton,artur,rissal,pia,florian")
    parser.add_argument("--random-character", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--frames", type=int, default=7200)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--episodes-eval", type=int, default=3)
    parser.add_argument("--reference-episodes", type=int, default=2)
    parser.add_argument("--log-every-episodes", type=int, default=10)
    parser.add_argument("--log-every-steps", type=int, default=1_000)
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON lines instead of Rich progress output")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--init-alpha", type=float, default=1.0)
    parser.add_argument("--min-alpha", type=float, default=0.0)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=10.0)
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
