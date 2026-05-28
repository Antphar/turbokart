# Turbo Kart Dash — RL Training Findings

*Last updated: 2026-05-27*

## Overview

We trained a DQN (Deep Q-Network) agent to play Turbo Kart Dash, a browser-based kart racing game rendered on HTML5 Canvas. Training runs entirely in a headless browser via Playwright, with a PyTorch DQN that exports weights as JSON for in-game inference at ~0.1 ms per forward pass.

The best agent beats the built-in waypoint AI on most tracks and uses items, drift, and ultimates.

---

## Architecture

| Component | Detail |
|---|---|
| Algorithm | DQN with target network, epsilon-greedy exploration |
| Network | MLP: input → 128 → 128 → 10 actions |
| Observation | 83 base features × 4 frame stack = **332 inputs** |
| Actions | 10 discrete (forward, turn L/R, brake, drift L/R, use item, use ultimate, forward+turn combos) |
| Replay buffer | 100k transitions, uniform sampling |
| Training | Playwright headless browser ↔ Python via `window.rlStep()` / `window.rlReset()` |
| Inference | JSON weights loaded in-browser, runs in requestAnimationFrame loop |

### Observation space (83 base features)

- **Navigation (4):** `headingError`, `nextHeadingError`, `targetDistance`, `nextTargetDistance`
- **Road raycasts (7):** -90° to +90° at 7 angles, normalized distance to road edge
- **Kart raycasts (7):** distances to nearest opponent kart per angle
- **Hazard raycasts (7):** distances to nearest hazard per angle
- **Pickup raycasts (7):** distances to nearest item pickup per angle
- **Booster raycasts (7):** distances to nearest booster pad per angle
- **Kart state (7):** `speed`, `forwardSpeed`, `lateralOffset`, `onRoad`, `driftCharge`, `boostActive`, `spinout`
- **Car stats (6):** `carMaxSpeed`, `carAcceleration`, `carGripNormal`, `carGripDrift`, `carTurnSpeed`, `carWeight`
- **Item/ultimate (13):** one-hot item type (10 item slots), `citationCount`, `ultimateCharge`, `shieldActive`
- **Status effects (7):** `invulnActive`, `doubleBlind`, `placeboSlow`, `mergePulling`, `mergeTethered`, `throttleLock`, `amended`

With frame-stack 4, the agent sees the current frame plus the 3 most recent, giving it implicit velocity/acceleration signals: **83 × 4 = 332 total inputs.**

---

## Key Findings

### 1. Frame skip was the single biggest improvement

| Config | Finish rate (solo, Core Mainframe) | Avg reward |
|---|---|---|
| frame-skip 1 (60 actions/sec) | ~10% | ~80 |
| frame-skip 4 (15 actions/sec) | ~50% | ~350 |
| **frame-skip 6 (10 actions/sec)** | **67%** | **499** |

At 60 Hz the agent made jittery micro-corrections that destabilized steering. At 10 Hz it learned smooth, deliberate control. Each decision covers ~6 physics frames, giving the kart time to respond before the next action.

### 2. Frame stacking provides velocity sensing

Without frame stacking the agent has no explicit velocity signal (it sees instantaneous distance, not delta-distance). Frame-stack 4 lets the model infer speed and trajectory from how observations change across frames.

SmoothGrad confirms this: `nextTargetDistance@-1` and `nextTargetDistance@-2` (target distance from 1 and 2 frames ago) have high attribution, meaning the model tracks how quickly it approaches waypoints.

### 3. Navigation dominates the learned policy

**Top 10 features by SmoothGrad attribution:**

| Feature | Attribution score |
|---|---|
| `nextHeadingError` | 57.1 |
| `headingError` | 55.0 |
| `nextTargetDistance` | 36.0 |
| `targetDistance` | 18.5 |
| `item:hotfix` | 14.1 |
| `mergeTethered` | 11.3 |
| `nextTargetDistance@-1` | 11.0 |
| `mergePulling` | 8.8 |
| `doubleBlind` | 8.2 |
| `invulnActive` | 8.0 |

The model is fundamentally a **heading-error minimizer** with reactive item/debuff handling. It steers toward the next checkpoint, adjusting for the one after that, while reacting to status effects that alter its control dynamics.

**Bottom 10 features (near-zero attention):**

| Feature | Attribution score |
|---|---|
| `throttleLock@-1` | 0.21 |
| `hazardRay15@-3` | 0.21 |
| `amended` | 0.23 |
| `throttleLock` | 0.23 |
| `pickupRay35@-3` | 0.23 |
| `pickupRay15@-3` | 0.25 |
| `amended@-3` | 0.25 |
| `pickupRay0@-3` | 0.25 |
| `pickupRay90@-3` | 0.26 |
| `pickupRay-60@-3` | 0.26 |

The oldest temporal frame (lag -3) of spatial raycasts is essentially ignored. The model doesn't need 4-frame history for spatial features — mainly for navigation scalars. `throttleLock` and `amended` debuffs are too rare to learn from at 300k steps.

### 4. The model learns item usage

`item:hotfix` (attribution 14.1) and `item:fasttrack` (5.0) are in the top 20 features. The agent has learned that holding a hotfix (heal) or fasttrack (speed boost) matters for decision-making. It activates items at contextually appropriate moments rather than immediately on pickup.

### 5. Per-track performance varies significantly

**Solo mode (no opponents):**

| Track | Finish rate | Avg reward | Avg laps |
|---|---|---|---|
| Core Mainframe | 67% | 499 | 2.0 |
| Compliance Chicane | **100%** | 499 | **3.0** |
| Black Ice Data Vault | 67% | 729 | 2.7 |
| Protocol Amendment Labyrinth | 33% | 440 | 2.0 |
| Audit Super Ring | 33% | 247 | 2.0 |

**Vs Classic waypoint AI (5 opponents):**

| Track | Finish rate | Avg reward | Player win rate |
|---|---|---|---|
| Core Mainframe | 67% | 667 | **100%** |
| Compliance Chicane | 0% | 188 | **100%** |
| Black Ice Data Vault | 0% | 472 | 67% |
| Protocol Amendment Labyrinth | 0% | 176 | **100%** |
| Audit Super Ring | 33% | 253 | 67% |

The agent dominates on Core Mainframe and holds a strong win rate across all tracks even when it doesn't finish the full 3 laps. Performance drops on tracks with complex geometry (Protocol Amendment Labyrinth has split-lane checkpoints) and high-speed curves (Audit Super Ring).

### 6. Self-play and items add training complexity but improve robustness

Training with `--with-opponents --with-items --self-play` forces the agent to handle collisions, item disruptions, and varied opponent strategies. The league system samples past checkpoints as opponents, weighted by recency and performance, creating a curriculum of increasing difficulty.

---

## Training configuration

**Best known command (now the defaults):**

```bash
uv run train_dqn.py --random-map --random-character --with-opponents --with-items --self-play
```

**Full parameter set:**

| Parameter | Value | Rationale |
|---|---|---|
| `--steps` | 300,000 | Enough for convergence on most tracks |
| `--frames` | 7,200 | 2 minutes of game time per episode at 60 FPS |
| `--frame-stack` | 4 | Velocity/acceleration sensing from observation history |
| `--frame-skip` | 6 | 10 actions/sec — smooth control, faster training |
| `--random-map` | enabled | Generalization across 5 tracks |
| `--random-character` | enabled | Generalization across car stats |
| `--with-opponents` | enabled | Learn racing behavior, not just driving |
| `--with-items` | enabled | Learn item usage and counterplay |
| `--self-play` | enabled | League of past checkpoints as opponents |
| `--league-limit` | 16 | Keep top 16 models in opponent pool |
| `--league-recency-tau` | 4.0 | Exponential recency weighting for opponent sampling |
| `--classic-opponent-prob` | 0.25 | 25% of training opponents are waypoint AI |
| `--eval-every` | 25,000 | Evaluate + checkpoint every 25k steps |
| `--episodes-eval` | 3 | 3 episodes per track per eval (solo + classic) |
| `--hidden` | 128 | Two hidden layers of 128 units |
| `--batch-size` | 256 | Replay buffer batch size |
| `--buffer-size` | 100,000 | Experience replay capacity |
| `--eps-decay` | 30,000 | Epsilon 1.0 → 0.05 over 30k steps |
| `--gamma` | 0.99 | Discount factor |
| `--lr` | 3e-4 | Adam learning rate |

---

## Model details

| Property | Value |
|---|---|
| Final model | `dqn-selfplay-booster-stack4-skip6.json` |
| File size | 1.8 MB (JSON with float32 weights) |
| Parameters | ~53k (83×4 → 128 → 128 → 10) |
| Inference time | <0.1 ms in browser |
| Training time | ~35 min for 300k steps on M-series Mac |
| Checkpoints saved | 12 (every 25k steps from 25k to 300k) |

---

## What didn't work well

1. **Frame-skip 1 (acting every frame):** Jittery steering, no coherent driving behavior.
2. **Frame-stack 1 (no history):** Agent couldn't sense velocity, drove erratically.
3. **Short training (50k steps):** Epsilon still >0.5, agent barely explores meaningful behavior.
4. **Solo-only training:** Agent learned to drive but couldn't handle opponent collisions.
5. **Single-map training:** Strong on one track, useless on others.

## What we'd try next

1. **PPO or SAC** for continuous action spaces (analog steering, variable throttle).
2. **Behavior cloning warm-start** from waypoint AI traces to skip early random exploration.
3. **Curriculum learning:** start solo on easy tracks, gradually add opponents/items/harder tracks.
4. **Larger networks or attention layers** to better exploit spatial raycast information.
5. **Reward shaping** for item usage — currently the agent discovers items through indirect reward.
6. **Native Python simulation** — extracting the physics engine from JS would eliminate Playwright overhead and enable ~10× faster training.

---

## Reproducibility

All training code is in `train_dqn.py` (single-file, `uv` script dependencies). Models export as JSON and load directly in the game browser. The manifest at `models/manifest.json` tracks all exported models with full metadata and eval results.

```bash
# Install and train
uv run train_dqn.py --random-map --random-character --with-opponents --with-items --self-play

# Evaluate
uv run eval_models.py --models all --html-report ghost_eval.html
```
