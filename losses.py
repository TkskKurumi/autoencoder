import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import keras.backend as K

def diff_var(y_true,y_pred):
	diff=y_true-y_pred
	ret=K.var(diff)
	return ret