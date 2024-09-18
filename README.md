🐭 Mouse Escape: Q-Learning Agent
This project demonstrates a Q-learning agent navigating a grid-based environment to reach its goal while avoiding obstacles like walls, "hell states," and teleportation tunnels. The agent learns to navigate efficiently using reinforcement learning by balancing exploration and exploitation with an epsilon-greedy policy.

🚀 Project Overview
Environment
The environment is designed as an 8x8 grid. The agent, represented as a male mouse, starts at the top-left corner of the grid (position (0, 0)), and its goal is to reach the female mouse located at (7, 7).

Obstacles
Walls: Impassable boundaries that block the agent's movement.
Hell States: Cells that apply a negative reward (-5) when entered.
Teleportation Tunnels: Special tunnels that instantly transport the agent between predefined entry and exit points, adding complexity to navigation.
Rewards
Positive Reward: The agent receives a reward of +10 for reaching the goal state.
Negative Reward: Entering a "hell state" applies a penalty of -5.
Living Penalty: The agent receives a small penalty (-0.01) for every move to encourage efficient pathfinding.
⚙️ Q-Learning Implementation
Algorithm
Q-learning is used to train the agent. The Q-learning algorithm updates its knowledge of the environment by estimating the optimal action-value function for each state-action pair based on rewards received from the environment.

Formula:

𝑄
(
𝑠
,
𝑎
)
=
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)=Q(s,a)+α[r+γ 
a
max
′
​
 Q(s 
′
 ,a 
′
 )−Q(s,a)]
Where:

𝛼
α is the learning rate.
𝛾
γ is the discount factor for future rewards.
𝑟
r is the reward for taking action 
𝑎
a in state 
𝑠
s.
𝑠
′
s 
′
  is the next state after action 
𝑎
a.
Exploration vs Exploitation
The agent uses an epsilon-greedy policy to balance exploration (trying new actions) and exploitation (selecting the action with the highest Q-value). Over time, epsilon decays, allowing the agent to exploit more learned knowledge and explore less.
🛠️ Program Structure
Key Functions
Step() Function: Handles the agent's movement and reward calculation based on the current action taken.
Reset() Function: Resets the agent’s position, reward accumulator, and step counter to restart the episode.
Actions
The agent can move in four directions:
Up: Decreases the y-coordinate.
Down: Increases the y-coordinate.
Left: Decreases the x-coordinate.
Right: Increases the x-coordinate.
📊 Challenges & Enhancements
Teleportation: Implementing teleportation tunnels added complexity, allowing the agent to instantly move between grid points. This required careful tuning to avoid infinite loops.
Exploration: Ensuring the agent explores enough to discover the best path while avoiding obstacles and dead ends was a key challenge.
Dynamic Obstacles: Walls and hell states required adaptive strategies to prevent the agent from getting stuck.
🔧 How to Run
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/mouse-escape-qlearning.git
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Run the agent simulation:
bash
Copy code
python main.py
📈 Future Work
Multi-agent Learning: Extending the project to include multiple agents with cooperative or competitive strategies.
Deep Q-Learning: Experimenting with deep learning methods to enhance the agent's decision-making.
