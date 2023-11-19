import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
# Disable eager execution to avoid issues with digital twin inference. 
tf.compat.v1.disable_eager_execution()

class Environment(gymnasium.Env):
	def __init__(self, digital_twin_path: str):
		self.dtwin = tf.keras.models.load_model(digital_twin_path)

		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)

		self.learning_history = pd.DataFrame({"input": [], "action": [], "reward": []})

	def reset(self, *, seed=None, options=None):
		self.observation = self.get_obs()
		info = {}

		return self.observation, info

	def step(self, action):
		reward = self.compute_reward(self.observation, action)

		terminated = True
		truncated = False
		info = {}

		new_record = pd.DataFrame({"input": [self.observation], 
							 "action": [action], 
							 "reward": [reward]}
							 )
		self.learning_history = pd.concat([self.learning_history, new_record], ignore_index=True)

		return self.observation, reward, terminated, truncated, info
	
	def render(self):
		self.learning_history.to_csv('learning_history.csv')
	
	def get_obs(self):
		obs = random.random()

		return np.array([obs])
	
	def compute_reward(self, obs, action):
		data_to_predict = pd.DataFrame({'input': [obs[0]], 'reaction': [action]})
		prediction = self.dtwin.predict(data_to_predict)[0][0]
		reward = 1.5 - abs(prediction - 1.5)
		
		return reward