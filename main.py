import numpy as np
import pandas as pd
import random

from system import System
from digital_twin import DigitalTwin
from rl_trainer import RLTrainer

if __name__ == "__main__":

	# Collect data from a running system.
	system = System()
	system.run(iters=1000)
	system_data = system.database.copy(deep=True)

	# Train a digital twin model to approximate the system's input-reaction-output' behavior.
	dtwin = DigitalTwin(data=system_data)
	dtwin.train_model(nb_epochs=50)

	# Train an RL agent using the digital twin of the system. 
	trainer = RLTrainer(digital_twin_path='digital_twin.h5')
	trainer.train(nb_train_steps=2)		
