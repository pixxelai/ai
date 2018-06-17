import numpy as np
from keras.models import Sequential
from keras import losses
from keras.callbacks import ModelCheckPoint
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import skimage

images = []
def data(file):
	for image in file:
		images.append(np.load(image))
	images = np.array(images)
	for i in range(len(images)):
		images[i] = np.array(images[i])
		images[i] = np.reshape(np.shape(images[i])[0]*np.shape(images[i])[1],1)
X = data(trainX)
Y = data(trainY)
#assume T is 30

model = Sequential()
model.add(LSTM(128,batch_input_shape = [None,30,288] , return_sequences = False))
model.add(Dropout(0.75))
model.add(Dense(256,activation = 'relu'))
model.add(Dense(1,activation = 'relu'))

model.summary()

model.compile(optimizer='rmsprop',loss=losses.squared_mean_error,metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath='best.hdf5', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=30, callbacks=callbacks_list, verbose=1)
