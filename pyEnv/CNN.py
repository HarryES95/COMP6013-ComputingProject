import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io as svio

def formatData(dataset_numpy):    
    data = np.full((len(dataset_numpy),20,64,64),[x for x in dataset_numpy])
    data = np.delete(data, 19, axis=1)    
    #Split data into pre and post frames / labels for those frames
    temp_dataset = np.full((len(data),10,64,64),[x[0:19:2] for x in data]) #Use Numpy 'fancy indexing' to select every other frame
    labels = np.full((len(data),9,64,64),[x[1:19:2] for x in data]) #Use Numpy 'fancy indexing' to select every other frame phase shifted +1
    #Stack dataset frames into shape (2,64,64)
    outer_temp = np.empty((1,9,2,64,64))
    for sequence in temp_dataset:
        count = 0
        temp = np.empty((1,2,64,64))
        while count != 9:
            temp = np.concatenate((temp,np.reshape(np.stack((sequence[count],sequence[count+1])),(1,2,64,64))))
            count += 1
        temp = np.delete(temp,0,axis=0)
        temp = np.reshape(temp,(1,9,2,64,64))
        outer_temp = np.concatenate((outer_temp,temp))
    dataset = np.delete(outer_temp,0,axis=0)
    return dataset, labels

# def loadDataset():
#     dataset = tfds.load(
#         name="moving_mnist", 
#         split="test", 
#         shuffle_files=False, 
#         data_dir="Dataset")
#     dataset = dataset.map(
#             lambda x: tf.squeeze(x["image_sequence"], 3) #change shape from (20,64,64,1) to (20,64,64)
#         ).map(
#             lambda y: tf.cast(y, dtype=tf.float16) #cast all tensors to float32 datatype
#         ).map(
#             lambda z: z/255. #normalise all values between 1 and 0.
#         )
#     #Convert to Numpy for better data manipulation
#     dataset_numpy = tfds.as_numpy(dataset)
#     dataset, labels = formatData(dataset_numpy)
#     dataset = np.reshape(dataset,(dataset.shape[0]*dataset.shape[1],2,64,64))
#     labels = np.reshape(labels,(labels.shape[0]*labels.shape[1],64,64))
#     dataset = tf.data.Dataset.from_tensor_slices(dataset)
#     labels = tf.data.Dataset.from_tensor_slices(labels)
#     train_length = len(dataset)*0.8
#     train_dataset = dataset.take(int(train_length))
#     train_labels = labels.take(int(train_length))
#     test_dataset = dataset.skip(int(train_length))
#     test_labels = labels.skip(int(train_length))
#     return train_dataset, test_dataset, train_labels, test_labels

def loadNumpyDataset():
    train_dataset = np.load("C:\\Users\\Harry\\Google Drive\\Computer Science\\COMP6013 - Computing Project\\Project Repository\\pyEnv\\Dataset\\train_dataset.npy")
    train_labels = np.load("Dataset\\train_labels.npy")
    test_dataset = np.load("Dataset\\test_dataset.npy")
    test_labels = np.load("Dataset\\test_labels.npy")
    return train_dataset, test_dataset, train_labels, test_labels

def enhance(sequence, model):
    sequence = np.reshape(sequence,(9,2,64,64))
    enhanced_sequence = np.insert(sequence,[1],[np.reshape(model.predict(np.reshape(x,(1,64,64,2))),(1,64,64)) for x in sequence],axis=1)
    return enhanced_sequence

train_dataset, test_dataset, train_labels, test_labels = loadNumpyDataset()

print(len(train_dataset))
print(len(train_labels))
print(len(test_dataset))
print(len(test_labels))

# model = models.Sequential()

# model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", input_shape=(64,64,2)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.MaxPool2D(pool_size=(2, 2), data_format="channels_last"))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.MaxPool2D(pool_size=(2, 2), data_format="channels_last"))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.MaxPool2D(pool_size=(2, 2), data_format="channels_last"))
# model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Concatenate()([model.get_layer('conv2d_9').output,model.get_layer('conv2d_5').output]))
# model.add(layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Concatenate()([model.get_layer('conv2d_11').output,model.get_layer('conv2d_3').output]))
# model.add(layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last"))
# model.add(layers.Concatenate()([model.get_layer('conv2d_13').output,model.get_layer('conv2d_1').output]))
# model.add(layers.Conv2D(1, (1, 1), activation='relu', padding='same', data_format="channels_last"))

inputs = layers.Input(shape=(64,64,2))
conv2d = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
conv2d_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv2d)
max_pooling2d = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_1)
conv2d_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(max_pooling2d)
conv2d_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_2)
max_pooling2d_1 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_3)
conv2d_4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(max_pooling2d_1)
conv2d_5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_4)
max_pooling2d_2 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_5)
conv2d_6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(max_pooling2d_2)
conv2d_7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2d_6)
max_pooling2d_3 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_7)
conv2d_8 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(max_pooling2d_3)
conv2d_9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv2d_8)
up_sampling2d = layers.Concatenate()([layers.UpSampling2D(size=(2, 2))(conv2d_9),conv2d_7])
conv2d_10 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up_sampling2d)
conv2d_11 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2d_10)
up_sampling2d_1 = layers.Concatenate()([layers.UpSampling2D(size=(2, 2))(conv2d_11),conv2d_5])
conv2d_12 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up_sampling2d_1)
conv2d_13 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_12)
up_sampling2d_2 = layers.Concatenate()([layers.UpSampling2D(size=(2,2))(conv2d_13),conv2d_3])
conv2d_14 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up_sampling2d_2)
conv2d_15 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_14)
up_sampling2d_3 = layers.Concatenate()([layers.UpSampling2D(size=(2,2))(conv2d_15),conv2d_1])
conv2d_16 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up_sampling2d_3)
conv2d_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2d_16)
outputs = layers.Conv2D(1, (1,1), activation='relu', padding='same')(conv2d_17)

model = models.Model(inputs=inputs, outputs=outputs, name="CNN")

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='huber')

# train_dataset = tf.stack([tf.reshape(tf.convert_to_tensor(y),(64,64,2)) for y in [x for x in train_dataset.as_numpy_iterator()]])
# train_labels = tf.stack([tf.reshape(tf.convert_to_tensor(y),(64,64)) for y in [x for x in train_labels.as_numpy_iterator()]])
# test_dataset = tf.stack([tf.reshape(tf.convert_to_tensor(y),(64,64,2)) for y in [x for x in test_dataset.as_numpy_iterator()]])
# test_labels = tf.stack([tf.reshape(tf.convert_to_tensor(y),(64,64)) for y in [x for x in test_labels.as_numpy_iterator()]])

history = model.fit(train_dataset,train_labels,batch_size=32,epochs=200)

original_sequence = test_dataset[0:9]
enhanced_sequence = enhance(original_sequence,model)

writer_original = svio.FFmpegWriter("original.mp4")
writer_enhanced = svio.FFmpegWriter("enhanced.mp4") 

original_sequence = np.reshape(original_sequence,(original_sequence.shape[0]*original_sequence.shape[-1],64,64))
enhanced_sequence = np.reshape(enhanced_sequence,(enhanced_sequence.shape[0]*enhanced_sequence.shape[1],64,64))

for frame in original_sequence:
    writer_original.writeFrame(frame*255)
writer_original.close()

for frame in enhanced_sequence:
    writer_enhanced.writeFrame(frame*255)
writer_enhanced.close()
