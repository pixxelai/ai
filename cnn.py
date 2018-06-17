
import numpy as np
import tensorflow as tf
from keras import losses
import skimage.transform
import skimage
from skimage import io

    
trainFiles=os.walk(trainpath)
testFiles=os.walk(testpath)
validFiles=os.walk(validpath)

'''target values ka code'''
def LoadImages(files):
    for t in files:
        images=[]
        tensor=[]
        #images.append(skimage.io.imread(t))
        images.append(np.load(t))
        images=np.array(images)
        tensor=images
        #tensor=np.expand_dims(images,axis=0)
        return images,tensor

trainingImages, trainTensors = LoadImages(trainFiles)
validationImages, validTensors = LoadImages(validFiles)
testImages, testTensors = LoadImages(testFiles)
train_targets=
valid_targets=
test_targets=


Print stats
print('Total number of test files is :',len(testFiles))
print('Total number of training files :',len(trainFiles))
print('Total number of validation files :', len(validFiles))



from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.
model.add(Conv2D(filters=128,kernel_size=3,strides=1,kernel_initializer='TruncatedNormal',padding='same',activation='relu',input_shape=(32,32,9)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128,kernel_size=3,strides=2,kernel_initializer='TruncatedNormal',padding='same',activation='relu'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256,kernel_size=3,strides=1,kernel_initializer='TruncatedNormal',padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256,kernel_size=3,strides=2,kernel_initializer='TruncatedNormal',padding='same',activation='relu'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=512,kernel_size=3,strides=1,kernel_initializer='TruncatedNormal',padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512,kernel_size=3,strides=1,kernel_initializer='TruncatedNormal',padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512,kernel_size=3,strides=2,kernel_initializer='TruncatedNormal',padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(optimizer='adam',loss=losses.mean_squared_error,metrics=['mse'])
## using checkpointer
from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='best_model_cnn.hdf5', verbose=1, save_best_only=True)
model.fit(trainTensors, train_targets, validation_data=(validTensors, validTargets),
       epochs=30, batch_size=1, callbacks=[checkpointer], verbose=1)


model.load_weights('best_model_cnn.hdf5')



#testImages, testTensors 
print(model.evaluate(testTensors,y_test))

