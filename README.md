# 🏎️ Turbo Kart Dash (Mainframe Edition)

[![HTML5](https://img.shields.io/badge/HTML5-supported-orange.svg?style=flat-square)](#)
[![Web Audio API](https://img.shields.io/badge/Web%20Audio-Procedural-blue.svg?style=flat-square)](#)
[![WebRTC P2P](https://img.shields.io/badge/WebRTC-Multiplayer-brightgreen.svg?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](#)

Welcome to **Turbo Kart Dash: Mainframe Edition**! This is a high-fidelity, premium retro arcade kart racing game built completely in single-file HTML5 with zero external assets, zero heavy framework dependencies, and zero pre-recorded audio files. 

Every single sound effect and music track is synthesized on-the-fly using the browser's native **Web Audio API**.

---

## 🚀 Key Features

### 🎧 Procedural Sound & Audio Sequencer
* **128-Step Synthwave Sequencer**: An 8-bar dynamic structured loop featuring an E Minor/C/D/Bm chord structure. 
* **Dynamic Arrangement sections**:
  * **Verse 1 & 2**: Smooth chiptune melodies over syncopated bass and standard drums.
  * **Chorus Climax**: High-energy arpeggios transposed up +1 octave with off-beat Hi-Hats and a driving four-on-the-floor kick!
  * **Breakdown/Tension**: Drums drop out as the synthesizer plays an ascending suspense chord progression while the sub-bass closes down with a sweep filter.
* **On-The-Fly Final Lap Acceleration**: Just like Mario Kart, entering the final lap dynamically accelerates the sequencer's base tempo from **124 BPM to 142 BPM** on-the-fly without altering the pitch or resetting the tracker.
* **Full Drum Synthesizer**: Procedural **Retro Kick Drums** (rapid swept sine waves) and **Retro Hi-Hats** (high-pass filtered noise) synthesized completely in Web Audio nodes.

### 🕹️ Authoritative Peer-to-Peer Multiplayer
* Powered by **PeerJS (WebRTC)**.
* Play with up to **4 online racers** over local networks or across the world with direct peer connections.
* Host-managed lobbies show joined players and let the host start the race when ready.
* Host-guest synchronization for local AI bots, item boxes, coin pickups, and hazards.
* Dynamic guest input and coordinate synchronization packets.
* Synchronized item activation events with spatialized projectile and shockwave feedback.

### 🚗 Advanced 2D Physics Engine
* Realistic vector-based drift mechanics with 3 distinct tiers of neon sparks (Blue -> Orange -> Neon Purple) based on drift duration.
* Wheel offset emitters placing drift sparks and smoke particles directly under the rear tires.
* Traction snapback speed boosts upon drift release.
* Elastic wall-bouncing physics with angle-based impact recoil.
* Slipstream drafting tunnels allowing you to lock onto karts ahead and gain overtaking turbo surges.
* Off-road digital static particles and spatialized tire noise when karts leave the racing surface.

### 🧨 Mainframe Items
* **Dossier Projectile**: Fire a bouncing audit packet at the kart ahead.
* **Merge Conflict**: Drop a red hazard block behind you.
* **De-auth Shockwave**: Trigger an expanding neon pulse that spins out nearby opponents.
* **Merge Request**: Tether to the kart ahead and pull yourself into review range.

### 🤖 Smart AI Autopilot Navigation
* Multi-point waypoint lookahead for proactive cornering.
* Adaptive cornering deceleration, strategic drift triggers, and automated obstacle avoidance raycasts (Left, Center, Right).
* strategic AI item decision engine and rubberbanding pace adjustments to keep the racing tight and competitive.

### 🌌 Cyberpunk Mainframe Visuals
* Neon vector grid lines replacing traditional grass.
* Pulsing glassmorphic holographic data pillars replacing standard trees.
* Glowing cyan boundaries with scrolling flow arrows on deep indigo highways.
* Glowing "FINISH!" banners and confetti explosions.
### 🗺️ Maps
* **Core Mainframe Circuit**: The classic test speedway.
* **Compliance Chicane**: Highly technical micro-circuit with acute 90° corners.
* **Audit Super Ring**: Giant wide-lane speed ring for massive drifts.
* **Black Ice Data Vault**: A complex vault-breach circuit with narrow corridors, switchbacks, and moving firewall gates.
* **Regulatory Dragon Run**: A long escape boulevard where a compliance dragon chases racers and fires audit projectiles.
* **Dragon's Escape** *(NEW)*: INFINITE SURVIVAL — a Japanese-themed open-ended trail. The dragon hunts from your left with progressive fire walls. One hit = dead. How far can you escape?

### 🎵 Dynamic Music Engine
* **5 Switchable Retro Tracks**: Mainframe Sunset, Neon Acid Overdrive, Retro Chiptune Horizon, Midnight House Circuit, Velvet Haze Crawl — each with unique tempo, groove, and instrumentation.
* **On-The-Fly Track Switching**: Change music style mid-race via the settings menu.
* **Japanese Audio for Dragon's Escape**: Procedural Taiko drums, Koto plucks, Shamisen strums, and Shakuhachi melodies that intensify from 90 BPM to 158 BPM as survival time increases.
* **Final Lap Tempo Surge**: Entering the final lap dynamically accelerates the sequencer's base tempo from **124 BPM to 142 BPM** without altering pitch or resetting the tracker.

---

## 🎮 How to Play

### Controls
* **Drive/Accelerate**: `W` or `Arrow Up`
* **Brake/Reverse**: `S` or `Arrow Down`
* **Steer**: `A`/`D` or `Arrow Left`/`Arrow Right`
* **Hop/Drift**: Hold `Space` while steering
* **Use Item**: `Left Shift` or `Right Shift`
* **Pause**: `P`
* **Mute/Unmute**: `M`
* **Restart**: `R`
* **CRT Overlay**: Toggle from the title menu

---

## 🛠️ Local Development & Running

Since the game is contained entirely in a single file (`index.html`), you can run it directly:
1. Double-click `index.html` in your file explorer to open it in any modern browser.
2. Alternatively, run a simple local server to enjoy instant loading and perfect WebRTC peer handshake stability:
   ```bash
   npx serve .
   ```
   or
   ```bash
   python3 -m http.server 8000
   ```

### 🤖 Training DQN AI Racers

Trained AI opponents are loaded from `models/manifest.json`. On GitHub Pages or a local HTTP server they appear automatically in the **AI Opponents** selector. If you open `index.html` directly with double-click, use **Import JSON** and select a model file manually.

**Quick start** — the defaults are tuned for a strong generalist agent:

```bash
uv run train_dqn.py --random-map --random-character --with-opponents --with-items --self-play
```

This trains for 300k steps across all 5 non-dragon maps with frame-stack 4 and frame-skip 6 (10 actions/sec). The current DQN defaults use the 15-action policy, pruned 216-feature observation stack, hidden width 64, Tanh activations, and orthogonal initialization. It evaluates solo and vs classic waypoint AI, prints SmoothGrad attribution plus action distribution, and saves checkpoints to `models/checkpoints/`.

**Full flags** for a named run:

```bash
uv run train_dqn.py \
  --steps 300000 \
  --random-map \
  --random-character \
  --frame-stack 4 \
  --frame-skip 6 \
  --hidden 64 \
  --activation tanh \
  --orthogonal-init \
  --with-opponents \
  --with-items \
  --self-play \
  --out models/dqn-my-agent.json \
  --model-id dqn-my-agent \
  --model-name "My Agent"
```

**Evaluate trained models** across maps with a leaderboard and ghost-path HTML report:

```bash
uv run eval_models.py --models all --html-report ghost_eval.html
```

Run randomized head-to-head races between two trained models:

```bash
uv run eval_head_to_head.py \
  --model-a dqn-dueling-pruned-300k \
  --model-b dqn-15act-h64-tanh-ortho-300k \
  --episodes 20
```

Head-to-head eval runs each matchup in both directions: model A as the player against model B opponents, then model B as the player against model A opponents. By default it keeps one classic waypoint AI in the opponent pack (`--classic-opponents 1`) so both models also face a baseline racer; use `--classic-opponents 0` for pure model-vs-model opponents. The report includes win rate, finish rate, reward, place, laps, citations, item uses, ultimate uses, and drift boosts.

The trainer prints model performance next to the classic waypoint AI reference with per-track solo/classic/win-rate splits, SmoothGrad input attribution, action distribution, gameplay counters, and character win distributions. Checkpoints are saved to `models/checkpoints/` and added to the manifest so future self-play runs can sample them as opponents.

**Key training findings:**
* Frame-skip 6 (10 actions/sec) was the single biggest improvement over frame-skip 1.
* Frame-stack 4 helps the model sense velocity and trajectory from observation history.
* The best current DQN variant uses 15 actions, hidden width 64, Tanh activations, and orthogonal initialization.
* The model primarily attends to heading error, target distance, and item status (hotfix/fasttrack).
* It learns to use drift and items strategically over ~150k steps.
* Protocol Amendment Labyrinth remains the hardest track due to split-lane checkpoints.

Future improvement idea: use behavior cloning from waypoint traces, then fine-tune with PPO or SAC for smoother steering and drift timing. DQN is intentionally the simple debug baseline.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
