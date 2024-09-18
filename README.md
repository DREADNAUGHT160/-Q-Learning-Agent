# 🐭 Mouse Escape: Q-Learning Agent

This project demonstrates a **Q-learning agent** navigating a grid-based environment to reach its goal while avoiding obstacles like walls, "hell states," and teleportation tunnels. The agent learns to navigate efficiently using **reinforcement learning** with an epsilon-greedy policy.

## 🚀 Overview

### Environment
- **8x8 grid**: Agent starts at `(0, 0)` and aims to reach the goal at `(7, 7)`.
- **Obstacles**: Walls (impassable), Hell States (negative rewards), and Teleportation Tunnels (instant movement).

### Rewards
- **+10** for reaching the goal.
- **-5** for entering Hell States.
- **-0.01** for every move (living penalty).

## ⚙️ Q-Learning

The agent uses Q-learning with the update formula:
Q(s, a) = Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

- **α (alpha)**: Learning rate.
- **γ (gamma)**: Discount factor.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation, with decaying epsilon.

### Actions
- **Move Up**: Decreases y-coordinate.
- **Move Down**: Increases y-coordinate.
- **Move Left**: Decreases x-coordinate.
- **Move Right**: Increases x-coordinate.

## 🛠️ How to Run



1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mouse-escape-qlearning.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the simulation:
    ```bash
    python main.py

  📈 Future Work
Multi-agent learning with cooperative or competitive strategies.
Implementing deep Q-learning to improve decision-making.
