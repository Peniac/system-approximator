import gymnasium as gym 
from gymnasium import spaces


class Environment(gym.Env):
	def __init__(self, digital_twin):
		self.dtwin = digital_twin

		self.action_space = spaces.Discrete(3)
		self.observation_space = 
		pass

	def reset(self):

		return observation, info
		pass

	def step(self):

		return observation, reward, terminated, False, info
		pass
	
	def render(self):
		pass

	def compute_reward(self):
		pass

	def get_obs(self):
		pass
	