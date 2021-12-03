import yield_data
import numpy as np
epoch=0
def train_ae(AE):
	global epoch
	data=yield_data.yield_data(512,128)
	data=np.array(data)
	AE.fit(x=[data],y=[data],batch_size=16,epochs=epoch+1,initial_epoch=epoch)
	epoch+=1