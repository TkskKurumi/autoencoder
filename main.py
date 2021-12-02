import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import keras.backend as K

import models,cv2,time
import yield_data,sys
import numpy as np
import models,training,losses

E,D,AE=models.ae()
opti=keras.optimizers.Nadam()
AE.compile(opti,loss=losses.diff_var)
while(True):
	training.train_ae(AE)
	models.save(E,D,AE)