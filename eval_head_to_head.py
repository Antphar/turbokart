# /// script
# dependencies = [
#   "numpy>=1.26",
#   "playwright>=1.40",
#   "rich>=13.0",
#   "torch>=2.2",
# ]
# ///
"""Run randomized head-to-head races between two Turbo Kart trained models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.table import Table

from rl_common import TurboKartEnv, launch_chromium, parse_csv


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("models", [])


def model_entry(manifest: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    for entry in manifest:
        if selector in {entry.get("id"), entry.get("name"), entry.get("path")}:
            return entry
    raise ValueError(f"Model not found in manifest: {selector}")


def load_model_payload(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = entry.get("path")
    if not path:
        raise ValueError(f"Manifest entry has no path: {entry.get('id')}")
    return json.loads((root / path).read_text(encoding="utf-8"))


def decide_action(page: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return page.evaluate(
        """(weights) => {
            const observation = getHeadlessObservation(game.player);
            const decision = weights.type === "sac"
                ? runHeadlessSac(weights, observation, game.player)
                : runHeadlessDqn(weights, observation, game.player);
            return decision.action;
        }""",
        payload,
    )


def ranking(page: Any) -> list[dict[str, Any]]:
    return page.evaluate(
        """() => rankAll().map(k => ({
            name: k.name,
            charId: k.charId,
            finished: !!k.finished,
            eliminated: !!k.eliminated,
            progress: progressValue(k),
            lap: k.lap,
            coins: k.coinsCollected || 0,
            itemUses: k.itemUseCount || 0,
            ultUses: k.ultUseCount || 0,
            driftBoosts: k.driftBoostCount || 0,
        }))"""
    )


def run_episode(
    env: TurboKartEnv,
    player_payload: dict[str, Any],
    opponent_payload: dict[str, Any],
    *,
    map_id: str,
    character: str,
    classic_opponents: int,
) -> dict[str, Any]:
    env.solo = False
    env.no_items = False
    env.no_hazards = True
    env.reset_with(
        map_id=map_id,
        character=character,
        opponent_models=[opponent_payload],
        classic_opponent_slots=classic_opponents,
    )
    done = False
    total_reward = 0.0
    info: dict[str, Any] = {}
    while not done:
        action = decide_action(env.page, player_payload)
        _, reward, done, info = env.step(action)
        total_reward += reward
    ranks = ranking(env.page)
    player_rank = next((idx + 1 for idx, kart in enumerate(ranks) if kart.get("charId") == character), None)
    player_stats = next((kart for kart in ranks if kart.get("charId") == character), {})
    return {
        "map": map_id,
        "character": character,
        "reward": total_reward,
        "finished": bool(info.get("finished")),
        "lap": float(info.get("lap", 0)),
        "progress": float(info.get("progress", 0)),
        "place": player_rank,
        "win": player_rank == 1,
        "coins": float(player_stats.get("coins", 0)),
        "itemUses": float(player_stats.get("itemUses", 0)),
        "ultUses": float(player_stats.get("ultUses", 0)),
        "driftBoosts": float(player_stats.get("driftBoosts", 0)),
        "winner": ranks[0].get("charId") if ranks else "unknown",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(1, len(rows))
    return {
        "episodes": len(rows),
        "win_rate": sum(1 for row in rows if row["win"]) / n,
        "finish_rate": sum(1 for row in rows if row["finished"]) / n,
        "avg_reward": sum(row["reward"] for row in rows) / n,
        "avg_place": sum(float(row["place"] or 9) for row in rows) / n,
        "avg_laps": sum(row["lap"] for row in rows) / n,
        "avg_progress": sum(row["progress"] for row in rows) / n,
        "avg_coins": sum(row["coins"] for row in rows) / n,
        "avg_items": sum(row["itemUses"] for row in rows) / n,
        "avg_ults": sum(row["ultUses"] for row in rows) / n,
        "avg_drift_boosts": sum(row["driftBoosts"] for row in rows) / n,
    }


def add_summary_row(table: Table, label: str, summary: dict[str, float]) -> None:
    table.add_row(
        label,
        str(int(summary["episodes"])),
        f"{summary['win_rate']:.2f}",
        f"{summary['finish_rate']:.2f}",
        f"{summary['avg_reward']:.1f}",
        f"{summary['avg_place']:.2f}",
        f"{summary['avg_laps']:.2f}",
        f"{summary['avg_progress']:.1f}",
        f"{summary['avg_coins']:.1f}",
        f"{summary['avg_items']:.1f}",
        f"{summary['avg_ults']:.1f}",
        f"{summary['avg_drift_boosts']:.1f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="index.html")
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frames", type=int, default=7200)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument(
        "--maps",
        default="core_mainframe,audit_super_ring,compliance_chicane,black_ice_data_vault,protocol_amendment_labyrinth",
    )
    parser.add_argument("--characters", default="anton,artur,rissal,pia,florian")
    parser.add_argument("--classic-opponents", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-auto-install-browser", dest="auto_install_browser", action="store_false")
    parser.set_defaults(auto_install_browser=True)
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(".").resolve()
    manifest = load_manifest(Path(args.manifest))
    entry_a = model_entry(manifest, args.model_a)
    entry_b = model_entry(manifest, args.model_b)
    payload_a = load_model_payload(root, entry_a)
    payload_b = load_model_payload(root, entry_b)
    maps = parse_csv(args.maps)
    characters = parse_csv(args.characters)
    console = Console()

    with sync_playwright() as p:
        browser = launch_chromium(p, args.auto_install_browser)
        page = browser.new_page()
        env = TurboKartEnv(
            page=page,
            index_path=Path(args.index),
            map_id=maps[0],
            character=characters[0],
            frames=args.frames,
            solo=False,
            no_items=False,
            no_hazards=True,
            frame_stack=4,
            frame_skip=args.frame_skip,
            classic_opponent_slots=args.classic_opponents,
        )
        env.load()
        rows_a: list[dict[str, Any]] = []
        rows_b: list[dict[str, Any]] = []
        for _ in range(args.episodes):
            map_id = random.choice(maps)
            char_a = random.choice(characters)
            rows_a.append(
                run_episode(
                    env,
                    payload_a,
                    payload_b,
                    map_id=map_id,
                    character=char_a,
                    classic_opponents=args.classic_opponents,
                )
            )
            char_b = random.choice([c for c in characters if c != char_a] or characters)
            rows_b.append(
                run_episode(
                    env,
                    payload_b,
                    payload_a,
                    map_id=map_id,
                    character=char_b,
                    classic_opponents=args.classic_opponents,
                )
            )
        browser.close()

    table = Table(title="TurboKart Head-to-Head")
    table.add_column("Player")
    table.add_column("N", justify="right")
    table.add_column("Win", justify="right")
    table.add_column("Finish", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Place", justify="right")
    table.add_column("Laps", justify="right")
    table.add_column("Progress", justify="right")
    table.add_column("Coins", justify="right")
    table.add_column("Items", justify="right")
    table.add_column("Ults", justify="right")
    table.add_column("Drifts", justify="right")
    add_summary_row(table, entry_a.get("name") or args.model_a, summarize(rows_a))
    add_summary_row(table, entry_b.get("name") or args.model_b, summarize(rows_b))
    console.print(table)


if __name__ == "__main__":
    main()
