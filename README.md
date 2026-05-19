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

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
