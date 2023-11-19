import numpy as np
import ray
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

from rl_env import Environment


class RLTrainer:
	def __init__(self, digital_twin_path:str):
		self.dtwin_path = digital_twin_path
		
		ray.init()
		assert ray.is_initialized()
		register_env("custom_env", lambda config: Environment(self.dtwin_path))

		self.trainer_config = {
			"env": "custom_env",
			"framework": "tf",
			"num_workers": 1,
			"num_gpus": 0
		}
		
		self.trainer = DQN(config=self.trainer_config)

	def train(self, nb_train_steps=10):

		for i in range(nb_train_steps):
			result = self.trainer.train()
			print(f"Iteration {i}, Mean Reward: {result['episode_reward_mean']}")

			print("\n Evaluating...")
			self.evaluate()

		ray.shutdown()
	
	def evaluate(self, nb_eval_steps=100):
		eval_env = Environment(self.dtwin_path)
		
		for j in range(nb_eval_steps):
			obs, _ = eval_env.reset()
			action = self.trainer.compute_single_action(obs, explore=False)
			_, reward, _, _, _ = eval_env.step(action)
			print(reward)
		
		eval_env.render()