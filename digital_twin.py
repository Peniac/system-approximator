import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

# from tensorflow import keras
# from tensorflow.keras import layers


class DigitalTwin:
	def __init__(self, data: pd.DataFrame):
		self.df = data
		_ = self._preprocess_data()

	def _preprocess_data(self):
		X = self.df[['input', 'reaction']]
		y = self.df['output']
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		self.scaler = StandardScaler()
		self.X_train_scaled = self.scaler.fit_transform(self.X_train)
		self.X_test_scaled = self.scaler.transform(self.X_test)

	def train_model(self, nb_epochs: int = 50):
		self.model = tf.keras.Sequential([
			tf.keras.layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(1)
		])

		self.model.compile(optimizer='adam', loss='mean_squared_error')

		self.model.fit(self.X_train_scaled, self.y_train, epochs=nb_epochs, batch_size=32, validation_split=0.2)

		mse = self.model.evaluate(self.X_test_scaled, self.y_test)
		print(f"Mean Squared Error on Test Set: {mse}")
		self.model.save("digital_twin.h5")

	def predict(self, new_data):
		new_data_norm = self.scaler.transform(new_data)
		prediction = self.model.predict(new_data_norm)

		return prediction