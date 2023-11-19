import numpy as np
import ray
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
from rl_env import Environment


class RLTrainer:
	def __init__(self, digital_twin_path:str):

		ray.init()
		assert ray.is_initialized()

		# Register the custom environment with Ray
		register_env("custom_env", lambda config: Environment(digital_twin_path))

		self.trainer_config = {
			"env": "custom_env",
			"framework": "tf",  # or "torch" for PyTorch
			"num_workers": 2,
			"num_gpus": 0,
			"num_cpus_per_worker": 1,
		}

	def train(self, train_steps=100):
		self.trainer = DQN(config=self.trainer_config)

		# Train the DQN agent for a number of iterations
		for i in range(train_steps):
			result = self.trainer.train()
			print(f"Iteration {i + 1}, Mean Reward: {result['episode_reward_mean']}")
		
		ray.shutdown()
	# def evaluate(eval_steps=100)