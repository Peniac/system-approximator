import random
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class System:
	def __init__(self):
		self.reactions = [0, 1, 2]
		self.database = pd.DataFrame({"input": [], "reaction": [], "output": []})

	def _get_input(self):
		x = random.random()

		return x

	def _react(self):
		a = random.choice(self.reactions)

		return a

	def _yield_output(self, x, a):
		y = x + a

		return y
	
	def run(self, iters: int = 10):

		for _ in range(iters):
			x = self._get_input()
			a = self._react()
			y = self._yield_output(x, a)
			new_record = pd.DataFrame({"input": [x], "reaction": [a], "output": [y]})
			self.database = pd.concat([self.database, new_record], ignore_index=True)
		
		self.database.to_csv("data.csv")
	
	def plot_data(self):
		df = self.database
		# Create a 3D scatter plot
		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(df['input'], df['reaction'], df['output'], c='blue', marker='o')

		ax.set_xlabel('Input')
		ax.set_ylabel('Reaction')
		ax.set_zlabel('Output')

		plt.title('Output as a function of Input and Reaction')

		plt.savefig('data.png')