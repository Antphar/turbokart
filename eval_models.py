# /// script
# dependencies = [
#   "playwright>=1.40",
#   "rich>=13.0",
# ]
# ///
"""Evaluate trained Turbo Kart DQN models and print a compact leaderboard.

Run:
  uv run eval_models.py --models all --maps core_mainframe,audit_super_ring
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.table import Table


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def launch_chromium(playwright: Any, auto_install: bool) -> Any:
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError as exc:
        message = str(exc)
        missing_browser = "Executable doesn't exist" in message or "playwright install" in message
        if not missing_browser or not auto_install:
            raise
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        return playwright.chromium.launch(headless=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("models", [])


def select_models(models: list[dict[str, Any]], selector: str) -> list[dict[str, Any]]:
    if selector == "all":
        return models
    wanted = set(parse_csv(selector))
    return [model for model in models if model.get("id") in wanted or model.get("name") in wanted]


def evaluate_model(page: Any, base_url: str, model_path: str, map_id: str, args: argparse.Namespace) -> dict[str, Any]:
    url = (
        base_url
        + "?headless=1"
        + "&agent=dqn"
        + f"&model={model_path}"
        + f"&map={map_id}"
        + f"&char={args.character}"
        + f"&frames={args.frames}"
        + f"&episodes={args.episodes}"
        + f"&solo={1 if args.solo else 0}"
        + f"&noItems={1 if args.no_items else 0}"
        + f"&noHazards={1 if args.no_hazards else 0}"
        + f"&frameSkip={args.frame_skip}"
        + ("&trace=1" if args.html_report else "")
        + f"&traceEvery={args.trace_every}"
    )
    page.goto(url, wait_until="load")
    error = page.evaluate("window.__HEADLESS_MODEL_ERROR__ || null")
    if error:
        raise RuntimeError(f"Model failed to load in browser: {error}")
    return page.evaluate("window.__HEADLESS_RESULT__")


class LocalServer:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.httpd: socketserver.TCPServer | None = None
        self.thread: threading.Thread | None = None
        self.url = ""

    def __enter__(self) -> str:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(self.root))
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{port}/index.html"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.url

    def __exit__(self, *args: Any) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def load_maps_from_index(index_path: Path) -> list[dict[str, Any]]:
    text = index_path.read_text(encoding="utf-8")
    start = text.find("const MAPS = [")
    if start == -1:
        return []
    start = text.find("[", start)
    depth = 0
    end = start
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    raw = text[start:end]
    # The map data is JavaScript, not JSON. This parser intentionally extracts only
    # simple fields needed for visualization.
    maps = []
    for block in raw.split("},"):
        if "name:" not in block or "waypoints:" not in block:
            continue
        name = extract_js_string(block, "name") or "Map"
        map_id = extract_js_string(block, "id") or name
        world_w = extract_js_number(block, "worldW") or 1
        world_h = extract_js_number(block, "worldH") or 1
        waypoints = []
        for part in block.split("{"):
            if "x:" in part and "y:" in part:
                x = extract_inline_number(part, "x")
                y = extract_inline_number(part, "y")
                if x is not None and y is not None:
                    waypoints.append({"x": x, "y": y})
        maps.append({"name": name, "id": map_id, "worldW": world_w, "worldH": world_h, "waypoints": waypoints})
    return maps


def extract_js_string(block: str, key: str) -> str | None:
    marker = f'{key}: "'
    pos = block.find(marker)
    if pos == -1:
        return None
    pos += len(marker)
    end = block.find('"', pos)
    return block[pos:end] if end != -1 else None


def extract_js_number(block: str, key: str) -> float | None:
    marker = f"{key}:"
    pos = block.find(marker)
    if pos == -1:
        return None
    return extract_inline_number(block[pos:], key)


def extract_inline_number(block: str, key: str) -> float | None:
    marker = f"{key}:"
    pos = block.find(marker)
    if pos == -1:
        return None
    pos += len(marker)
    chars = []
    while pos < len(block) and block[pos] in " \t":
        pos += 1
    while pos < len(block) and (block[pos].isdigit() or block[pos] in ".-"):
        chars.append(block[pos])
        pos += 1
    try:
        return float("".join(chars))
    except ValueError:
        return None


def scale_points(points: list[dict[str, float]], width: float, height: float, pad: float = 18) -> list[tuple[float, float]]:
    if not points:
        return []
    min_x = min(p["x"] for p in points)
    max_x = max(p["x"] for p in points)
    min_y = min(p["y"] for p in points)
    max_y = max(p["y"] for p in points)
    scale = min((width - pad * 2) / max(1, max_x - min_x), (height - pad * 2) / max(1, max_y - min_y))
    off_x = (width - (max_x - min_x) * scale) / 2
    off_y = (height - (max_y - min_y) * scale) / 2
    return [(off_x + (p["x"] - min_x) * scale, off_y + (p["y"] - min_y) * scale) for p in points]


def render_html_report(results: list[dict[str, Any]], maps: dict[str, dict[str, Any]], out_path: Path) -> None:
    cards = []
    colors = ["#57f2ff", "#fd9927", "#a4ff80", "#ff4d6d", "#bd57ff", "#ffd86b"]
    for idx, item in enumerate(results):
        result = item["result"]
        map_data = maps.get(item["map"], {})
        waypoints = map_data.get("waypoints", [])
        trace = result.get("trace") or []
        width, height = 460, 280
        track_points = scale_points(waypoints, width, height)
        trace_points = scale_points([{"x": p["x"], "y": p["y"]} for p in trace], width, height)
        track_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in track_points)
        trace_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in trace_points)
        color = colors[idx % len(colors)]
        ghosts = []
        if trace_points:
            step = max(1, len(trace_points) // 18)
            for gi, (x, y) in enumerate(trace_points[::step]):
                ghosts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" opacity="{0.25 + gi / max(1, len(trace_points[::step])) * 0.65:.2f}"/>')
        cards.append(
            f"""
            <section class="card">
              <h2>{item['model']} · {item['map']}</h2>
              <p>finish {result.get('aggregate', {}).get('finishCount', 0)}/{result.get('config', {}).get('episodes', 1)}
                 · reward {result.get('aggregate', {}).get('avgReward', 0):.1f}
                 · laps {result.get('aggregate', {}).get('totalPlayerLaps', 0)}</p>
              <svg viewBox="0 0 {width} {height}">
                <rect width="{width}" height="{height}" rx="14" fill="#08061a"/>
                <polyline points="{track_poly}" fill="none" stroke="rgba(255,255,255,.20)" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
                <polyline points="{track_poly}" fill="none" stroke="#7b75ff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
                <polyline points="{trace_poly}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                {''.join(ghosts)}
              </svg>
            </section>
            """
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TurboKart Ghost Eval</title>
<style>
body {{ margin: 0; padding: 24px; background: #060514; color: #fff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }}
.card {{ background: rgba(13,11,33,.78); border: 1px solid rgba(87,242,255,.18); border-radius: 18px; padding: 16px; box-shadow: 0 12px 32px rgba(0,0,0,.35); }}
h1 {{ margin-top: 0; }}
h2 {{ font-size: 16px; margin: 0 0 6px; }}
p {{ color: #a8acd0; margin: 0 0 12px; font-size: 13px; }}
svg {{ width: 100%; height: auto; display: block; }}
</style></head><body><h1>TurboKart Ghost Evaluation</h1><div class="grid">{''.join(cards)}</div></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="index.html")
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--models", default="all", help="'all' or comma-separated ids/names")
    parser.add_argument("--maps", default="core_mainframe,audit_super_ring,compliance_chicane")
    parser.add_argument("--character", default="florian")
    parser.add_argument("--frames", type=int, default=7200)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--solo", action="store_true", default=True)
    parser.add_argument("--with-opponents", dest="solo", action="store_false")
    parser.add_argument("--no-items", action="store_true", default=True)
    parser.add_argument("--with-items", dest="no_items", action="store_false")
    parser.add_argument("--no-hazards", action="store_true", default=True)
    parser.add_argument("--with-hazards", dest="no_hazards", action="store_false")
    parser.add_argument("--html-report", default=None, help="Write an HTML ghost-path visualization report")
    parser.add_argument("--trace-every", type=int, default=20)
    parser.add_argument("--no-auto-install-browser", dest="auto_install_browser", action="store_false")
    parser.set_defaults(auto_install_browser=True)
    args = parser.parse_args()

    root = Path(".").resolve()
    manifest_path = Path(args.manifest)
    models = select_models(load_manifest(manifest_path), args.models)
    maps = parse_csv(args.maps)
    console = Console()

    table = Table(title="TurboKart Model Evaluation")
    table.add_column("Model", style="cyan")
    table.add_column("Map", style="magenta")
    table.add_column("Finish", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Laps", justify="right")
    table.add_column("Progress", justify="right")
    report_results = []
    map_lookup = {m["id"]: m for m in load_maps_from_index(Path(args.index))}

    with LocalServer(root) as base_url, sync_playwright() as p:
        browser = launch_chromium(p, args.auto_install_browser)
        page = browser.new_page()
        for model in models:
            model_path = model.get("path")
            if not model_path:
                continue
            for map_id in maps:
                result = evaluate_model(page, base_url, model_path, map_id, args)
                report_results.append(
                    {
                        "model": model.get("name") or model.get("id") or model_path,
                        "map": map_id,
                        "result": result,
                    }
                )
                agg = result.get("aggregate", {})
                table.add_row(
                    model.get("name") or model.get("id") or model_path,
                    map_id,
                    f"{agg.get('finishCount', 0) / max(1, args.episodes):.2f}",
                    f"{agg.get('avgReward', 0):.1f}",
                    f"{agg.get('totalPlayerLaps', 0) / max(1, args.episodes):.2f}",
                    f"{agg.get('avgPlayerProgress', 0):.1f}",
                )
        browser.close()

    console.print(table)
    if args.html_report:
        out_path = Path(args.html_report)
        render_html_report(report_results, map_lookup, out_path)
        console.print(f"[green]Wrote ghost report:[/green] {out_path}")


if __name__ == "__main__":
    main()
