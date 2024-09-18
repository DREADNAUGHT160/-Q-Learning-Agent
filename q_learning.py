import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, q_table_save_path="q_table.npy"):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = tuple(next_state)
            total_reward += reward

            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    np.save(q_table_save_path, q_table)
    print("Training finished and Q-table saved.")

def visualize_q_table(hell_state_coordinates, goal_coordinates, wall_state_coordinates, teleportation_tunnels, actions=["Up", "Down", "Right", "Left"], q_values_path="q_table.npy"):
    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ['blue', 'purple', 'orange', 'green', 'red']  # Define a list of colors for teleportation tunnels
        color_index = 0  # Initialize color index

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for hell in hell_state_coordinates:
                mask[hell] = True
            for wall in wall_state_coordinates:
                mask[wall] = True
            for start, end in teleportation_tunnels.items():
                mask[start] = True
                mask[end] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green', ha='center', va='center', weight='bold', fontsize=14)
            for hell in hell_state_coordinates:
                ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=14)
            for wall in wall_state_coordinates:
                ax.text(wall[1] + 0.5, wall[0] + 0.5, 'W', color='black', ha='center', va='center', weight='bold', fontsize=14)
            
            for (start, end) in teleportation_tunnels.items():
                color = colors[color_index % len(colors)]  # Rotate through the list of colors
                ax.text(start[1] + 0.5, start[0] + 0.5, 'S', color=color, ha='center', va='center', weight='bold', fontsize=14)
                ax.text(end[1] + 0.5, end[0] + 0.5, 'E', color=color, ha='center', va='center', weight='bold', fontsize=14)
                color_index += 1  # Move to the next color

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
