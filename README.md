# 🐭 Mouse Escape: Q-Learning Agent

This project is now a **fully client-side simulation** that can be deployed as-is to
[Netlify](https://www.netlify.com/) or any static hosting provider. A JavaScript
implementation of the Q-learning agent trains entirely in the browser, so no
backend services are required.

## 🚀 Features

- **Interactive training controls** – configure episodes, learning rate, discount
  factor, and epsilon schedule.
- **Live environment visualization** – watch the 8×8 maze update as the agent
  explores, including walls, hell states, and teleporters.
- **Reward analytics** – a Chart.js line graph and summary stats track the
  agent’s performance during training.
- **Netlify-ready** – static assets (`index.html`, `style.css`, `script.js`) and a
  minimal [`netlify.toml`](netlify.toml) publish configuration.

## 🧠 Environment Overview

- **Grid:** 8×8, with the mouse starting at `(0, 0)` and the goal at `(7, 7)`.
- **Obstacles:**
  - Walls (impassable)
  - Hell states (−5 reward, terminal)
  - Teleporters (instant relocation)
- **Rewards:**
  - `+10` for reaching the goal
  - `−5` for entering a hell state
  - `−0.01` living penalty per move
- **Actions:** up, down, left, right
- **Policy:** epsilon-greedy with configurable decay

## 🛠️ Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mouse-escape-qlearning.git
   cd mouse-escape-qlearning
   ```
2. Start a local static server (any option works). For example, using Python:
   ```bash
   python -m http.server 8000
   ```
3. Visit the site:
   ```bash
   http://localhost:8000/
   ```

Alternatively, open `index.html` directly in your browser.

## 🌐 Deploying to Netlify

1. [Create a new site from Git](https://app.netlify.com/start) and select this
   repository.
2. Use the defaults:
   - **Build command:** leave empty (the project is prebuilt)
   - **Publish directory:** `.`
3. Deploy. Netlify will serve the static files immediately, and the Q-learning
   simulation will run entirely in the visitor’s browser.

## 📈 Future Work

- Multi-agent learning with cooperative or competitive strategies
- Deep Q-learning variations for richer state representations
- Persistent storage for Q-tables across sessions via IndexedDB or cloud sync
