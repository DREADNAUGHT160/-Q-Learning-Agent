from MouseEscape_env import MouseEscape
from q_learning import train_q_learning, visualize_q_table

hell_states_list = [(4, 1), (2, 0), (2, 2), (1, 4), (4, 3), (6, 3), (5, 6)]
wall_states_list = [(5, 1), (6, 1), (7, 1), (6, 4), (6, 5), (6, 6), (4, 6), (3, 6), (2, 6), (1, 7),
                    (1, 6), (1, 5), (1, 2), (0, 2), (1, 0), (0, 0), (4, 4)]

teleportation_tunnels = {
    (4, 2): (2, 5),
    (0, 1): (5, 5),
    (0, 7): (5, 7)
}

env = MouseEscape(grid_size=8,
                  bg_image_path='background_snake.png',
                  agent_image_path='agent.png',
                  goal_image_path='goal_image.png',
                  portal_image_path='black_hole.png',
                  agent_at_goal_image_path='reach_goal_image.png',
                  hell_states=hell_states_list,
                  wall_states=wall_states_list)

train_q_learning(env, no_episodes=300, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, alpha=0.1, gamma=0.99)
visualize_q_table(hell_states_list, (env.grid_size-1, env.grid_size-1), wall_states_list, teleportation_tunnels)
