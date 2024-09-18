import numpy as np
import pygame
import gymnasium as gym

class MouseEscape(gym.Env):
    def __init__(self, grid_size=8, bg_image_path=None, agent_image_path=None, goal_image_path=None, portal_image_path=None, agent_at_goal_image_path=None, hell_states=None, wall_states=None) -> None:
        super(MouseEscape, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array([grid_size-1, grid_size-1])
        self.done = False
        self.hell_states = [np.array(hell) for hell in hell_states] if hell_states else []
        self.wall_states = [np.array(wall) for wall in wall_states] if wall_states else []
        self.teleportation_tunnels = {
            (4, 2): (2, 5),
            (0, 1): (5, 5),
            (0, 7): (5, 7)
        }

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        
        # Load images:
        self.bg_image = None
        if bg_image_path:
            self.bg_image = pygame.image.load(bg_image_path)
            self.bg_image = pygame.transform.scale(self.bg_image, (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        
        self.agent_image = None
        if agent_image_path:
            self.agent_image = pygame.image.load(agent_image_path)
            self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        
        self.goal_image = None
        if goal_image_path:
            self.goal_image = pygame.image.load(goal_image_path)
            self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))
        
        self.portal_image = None
        if portal_image_path:
            self.portal_image = pygame.image.load(portal_image_path)
            self.portal_image = pygame.transform.scale(self.portal_image, (self.cell_size, self.cell_size))
        
        self.agent_image_at_goal = None
        if agent_at_goal_image_path:
            self.agent_image_at_goal = pygame.image.load(agent_at_goal_image_path)
            self.agent_image_at_goal = pygame.transform.scale(self.agent_image_at_goal, (self.cell_size, self.cell_size))

    def reset(self):
        self.state = np.array([7, 0])
        self.done = False
        self.reward = 0
        return self.state, self.info

    def step(self, action):
        if action == 0:  # Up
            next_state = self.state + np.array([-1, 0])
        elif action == 1:  # Down
            next_state = self.state + np.array([1, 0])
        elif action == 2:  # Right
            next_state = self.state + np.array([0, 1])
        elif action == 3:  # Left
            next_state = self.state + np.array([0, -1])

        if (next_state < 0).any() or (next_state >= self.grid_size).any() or any(np.array_equal(next_state, wall) for wall in self.wall_states):
            next_state = self.state

        self.state = next_state

        if any(np.array_equal(self.state, hell) for hell in self.hell_states):
            self.reward = -100
            self.done = True
        elif np.array_equal(self.state, self.goal):
            self.reward = 100
            self.done = True
        elif tuple(self.state) in self.teleportation_tunnels:
            self.reward = 5 if np.array_equal(self.state, [0, 7]) else -1
            self.state = np.array(self.teleportation_tunnels[tuple(self.state)])
        else:
            self.reward = -1

        return self.state, self.reward, self.done, self.info

    def render(self):
        if self.bg_image:
            self.screen.blit(self.bg_image, (0, 0))
        else:
            self.screen.fill((255, 255, 255))

        if self.goal_image:
            self.screen.blit(self.goal_image, (self.goal[1]*self.cell_size, self.goal[0]*self.cell_size))

        if self.portal_image:
            for tunnel in self.teleportation_tunnels:
                self.screen.blit(self.portal_image, (tunnel[1]*self.cell_size, tunnel[0]*self.cell_size))

        if self.agent_image:
            self.screen.blit(self.agent_image, (self.state[1]*self.cell_size, self.state[0]*self.cell_size))

        pygame.display.flip()

    def close(self):
        pygame.quit()
