
import tensorflow as tf
from tfHelper import tfHelper
import numpy as np

# model = tfHelper.load_model("model")

######################### Predict #########################

def predict(model, x_pred):
	predictions = model.predict(x_pred)

	AllPrediction = []
	for i in range(predictions.shape[0]):
		indexMax = np.argmax(predictions[i])
		AllPrediction.append(indexMax)
	return AllPrediction
