from system import System
from digital_twin import DigitalTwin
from rl_trainer import RLTrainer

import numpy as np
import pandas as pd
import random

if __name__ == "__main__":
	system = System()

	system.run(iters=1000)

	system_data = system.database.copy(deep=True)
	dt = DigitalTwin(data=system_data)
	dt.train_model(nb_epochs=10)

	trainer = RLTrainer(digital_twin_path='digital_twin.h5')
	trainer.train()
	

	# for _ in range(10):
	# 	data_to_predict = pd.DataFrame({'input': [system._get_input()], 'reaction': [system._react()]})
	# 	prediction = dt.predict(new_data=data_to_predict)
	# 	print(f'Data: {data_to_predict} \n Prediction: {prediction} \n\n')

	
		