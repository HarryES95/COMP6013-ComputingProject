import tensorflow as tf
from tensorflow.keras import layers,models
import skvideo.io as svio
import matplotlib.pyplot as plt
import numpy as np

def loadVideo():
    numpy_video = svio.vread("Dataset\\Charlie Chaplin_ Easy Street.mp4", as_grey=True)
    return numpy_video

def formatData(dataset_numpy):    
    #Split data into pre and post frames / labels for those frames
    temp_dataset = dataset_numpy[0:len(dataset_numpy):2] #Use Numpy 'fancy indexing' to select every other frame
    labels = dataset_numpy[1:len(dataset_numpy):2] #Use Numpy 'fancy indexing' to select every other frame phase shifted +1
    #Stack dataset frames into shape (2,64,64)
    count = 0
    temp = np.empty((1,2,dataset_numpy.shape[-3],dataset_numpy.shape[-2]))
    while count != len(temp_dataset):
        temp = np.concatenate((temp,np.reshape(np.stack((temp_dataset[count],temp_dataset[count+1])),(1,2,dataset_numpy.shape[-3],dataset_numpy.shape[-2]))))
        count += 1
    dataset = np.delete(temp,0,axis=0)
    return dataset, labels

numpy_video = loadVideo()
train_dataset,train_labels = formatData(numpy_video)

inputs = layers.Input(shape=(numpy_video.shape[1],numpy_video.shape[2],2))
conv2d = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
conv2d_1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(conv2d)
max_pooling2d = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_1)
conv2d_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(max_pooling2d)
conv2d_3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2d_2)
max_pooling2d_1 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_3)
conv2d_4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(max_pooling2d_1)
conv2d_5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_4)
max_pooling2d_2 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_5)
conv2d_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(max_pooling2d_2)
conv2d_7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_6)
up_sampling2d = layers.Concatenate()([layers.UpSampling2D(size=(2, 2))(conv2d_7),conv2d_5])
conv2d_8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up_sampling2d)
conv2d_9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_8)
up_sampling2d_1 = layers.Concatenate()([layers.UpSampling2D(size=(2,2))(conv2d_9),conv2d_3])
conv2d_10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up_sampling2d_1)
conv2d_11 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2d_10)
up_sampling2d_2 = layers.Concatenate()([layers.UpSampling2D(size=(2,2))(conv2d_11),conv2d_1])
conv2d_12 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up_sampling2d_2)
conv2d_13 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2d_12)
outputs = layers.Conv2D(1, (1,1), activation='relu', padding='same')(conv2d_13)

model = models.Model(inputs=inputs, outputs=outputs, name="CNN")

model.compile(optimizer='adam',
              loss='huber')

history = model.fit(train_dataset,train_labels,batch_size=32,epochs=100)

print("pause")